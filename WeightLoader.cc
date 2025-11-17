// -*- c++ -*-

#include <sst/core/sst_config.h>
#include "WeightLoader.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <inttypes.h>

using namespace SST;
using namespace SST::SnnDL;

WeightLoader::WeightLoader(ComponentId_t id, Params& params)
    : Component(id), output_(nullptr), memory_(nullptr), loaded_(false) {
    verbose_ = params.find<int>("verbose", 0);
    weight_file_ = params.find<std::string>("weight_file", "");
    base_addr_start_ = params.find<uint64_t>("base_addr_start", 0);
    per_core_stride_ = params.find<uint64_t>("per_core_stride", 0);
    num_cores_ = params.find<int>("num_cores", 1);
    neurons_per_core_ = params.find<uint32_t>("neurons_per_core", 64);
    // 行/列可选参数：未提供或为0时退回到neurons_per_core_
    rows_per_core_ = params.find<uint32_t>("rows_per_core", 0);
    cols_per_core_ = params.find<uint32_t>("cols_per_core", 0);
    if (rows_per_core_ == 0) rows_per_core_ = neurons_per_core_;
    if (cols_per_core_ == 0) cols_per_core_ = neurons_per_core_;
    fill_value_ = params.find<float>("fill_value", 0.5f);
    weight_format_ = params.find<std::string>("weight_format", "bin");
    raw_mode_ = (weight_format_ == "raw");
    bcsr_enable_ = params.find<int>("bcsr_enable", 0) != 0 || (weight_format_ == "bcsr");
    bcsr_br_ = params.find<uint32_t>("bcsr_block_rows", 16);
    bcsr_bc_ = params.find<uint32_t>("bcsr_block_cols", 16);
    bcsr_val_bytes_ = params.find<uint32_t>("bcsr_val_bytes", 4);
    bcsr_idx_bytes_ = params.find<uint32_t>("bcsr_idx_bytes", 2);
    bcsr_pattern_ = params.find<std::string>("bcsr_pattern", "diag");
    per_core_files_ = params.find<int>("per_core_files", 0) != 0;
    file_template_ = params.find<std::string>("file_template", "");
    single_file_ = params.find<std::string>("single_file", "");
    row_major_ = params.find<int>("row_major", 1) != 0;
    chunk_size_bytes_ = params.find<uint32_t>("chunk_size_bytes", 64);
    validate_length_ = params.find<int>("validate_length", 1) != 0;
    file_core_offset_ = params.find<int>("file_core_offset", 0);
    timed_seed_enable_ = params.find<int>("timed_seed_enable", 1) != 0;
    timed_seed_count_ = params.find<uint32_t>("timed_seed_count", 1);
    timed_seed_allow_cache_ = params.find<int>("timed_seed_allow_cache", 0) != 0;
    timed_write_window_ = std::max<uint32_t>(timed_seed_count_, 256u);
    loader_done_key_ = params.find<std::string>("loader_done_key", "");

    output_ = new Output("WeightLoader[@p:@l]: ", verbose_, 0, Output::STDOUT);
    output_->verbose(CALL_INFO, 1, 0, "🔧 初始化WeightLoader\n");

    if (!loader_done_key_.empty()) {
        loader_done_shared_.initialize(loader_done_key_, 1, 0);
        loader_done_shared_initialized_ = true;
    }

    // 注册统计（构造期，满足CSV注册时机约束）
    stat_bytes_total_ = registerStatistic<uint64_t>("weight_bytes_written_total");
    stat_chunks_total_ = registerStatistic<uint64_t>("weight_write_chunks_total");
    stat_chunk_bytes_hist_ = registerStatistic<uint64_t>("weight_write_chunk_bytes");
    stat_write_latency_cycles_hist_ = registerStatistic<uint64_t>("weight_write_latency_cycles");
    stat_write_total_cycles_ = registerStatistic<uint64_t>("weight_write_total_cycles");

    // 加载 StandardMem 子组件
    memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memory", ComponentInfo::SHARE_NONE,
        registerTimeBase("1ns"),
        new SST::Interfaces::StandardMem::Handler2<WeightLoader, &WeightLoader::handleMemoryResponse>(this));
    if (!memory_) {
        output_->fatal(CALL_INFO, -1, "❌ WeightLoader未配置StandardMem子组件\n");
    }
    // 规范化/健壮性检查
    normalizeParams_();
}

WeightLoader::~WeightLoader() {
    delete output_;
}

void WeightLoader::init(unsigned int phase) {
    // 将init相位转发给StandardMem
    if (memory_) memory_->init(phase);

    if (phase == 0) {
        if (output_) {
            output_->verbose(CALL_INFO, 1, 0,
                "[WL-init] loader_done_key=%s timed=%d allow_cache=%d base=0x%llx stride=%" PRIu64 " cores=%d\n",
                loader_done_key_.c_str(),
                timed_seed_enable_ ? 1 : 0,
                timed_seed_allow_cache_ ? 1 : 0,
                (unsigned long long)base_addr_start_, per_core_stride_, num_cores_);
        }
    }
    if (phase == 1 && !loaded_) {
        if (timed_seed_enable_ && timed_seed_allow_cache_) {
            runtime_load_needed_ = true;
        } else {
            loadFileOnce();
            publishLoaderDone_();
        }
    }
}

void WeightLoader::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader setup完成\n");
    
    // 若未启用 timed，则在 setup 阶段直接进行一次性加载并发布完成信号，
    // 可确保上游在仿真开始前即可看到 loader_done=1（适配短时快测）。
    if (!timed_seed_enable_) {
        if (!loaded_) {
            loadFileOnce();
        }
        publishLoaderDone_();
        runtime_load_needed_ = false;
    } else {
        // 启用 timed seed：注册时钟并由运行时驱动加载，完成时在写响应路径发布 loader_done
        if (!clock_registered_) {
            // Use Handler2 (supports checkpointing) to silence deprecation warning
            registerClock("1GHz", new SST::Clock::Handler2<WeightLoader, &WeightLoader::onClockTick>(this));
            clock_registered_ = true;
        }
        // 在setup完成后的第一个时钟周期进行权重加载
        runtime_load_needed_ = true;
    }
}

void WeightLoader::finish() {
    // output_->verbose(CALL_INFO, 1, 0, "🏁 WeightLoader 完成\n");
}

bool WeightLoader::onClockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;
    
    // 在运行时第一个周期进行权重加载
    if (runtime_load_needed_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔄 在运行时第一个周期进行权重加载\n");
        runtime_load_needed_ = false;
        loaded_ = false;  // 重置标志以允许重新加载
        if (!timed_seed_allow_cache_) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ timed_seed_allow_cache=0，回退为untimed写入以避免缓存coherence拒绝Write\n");
            loadFileOnce();
        } else {
            // 标记timed写入统计起点
            write_started_timed_ = true;
            write_first_issue_cycle_ = current_cycle_;
            loadFileOnceRuntime();  // 使用运行时加载函数
        }
    }
    driveTimedJobs_();
    return false;
}
void WeightLoader::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
    if (!req) return;
    
    // 跟踪写入完成
    if (pending_writes_ > 0) {
        pending_writes_--;
        output_->verbose(CALL_INFO, 3, 0, "📝 写入响应收到，剩余待处理: %u\n", pending_writes_);
        
        // 检查是否所有写入都已完成
        if (pending_writes_ == 0 && !all_writes_completed_) {
            all_writes_completed_ = true;
            // output_->verbose(CALL_INFO, 1, 0, "🎉 所有权重写入操作已完成！\n");
            if (write_started_timed_ && stat_write_total_cycles_) {
                uint64_t total = (current_cycle_ >= write_first_issue_cycle_) ? (current_cycle_ - write_first_issue_cycle_) : 0;
                stat_write_total_cycles_->addData(total);
            }
            publishLoaderDone_();
        }
    }
    
    // 仅timed写入：记录单次写入端到端时延
    if (stat_write_latency_cycles_hist_) {
        auto it = write_issue_cycle_.find(req->getID());
        if (it != write_issue_cycle_.end()) {
            uint64_t start = it->second;
            uint64_t lat = (current_cycle_ >= start) ? (current_cycle_ - start) : 0;
            stat_write_latency_cycles_hist_->addData(lat);
            write_issue_cycle_.erase(it);
        }
    }

    delete req;
    driveTimedJobs_();
}


void WeightLoader::normalizeParams_() {
    // 规范化行/列参数
    if (rows_per_core_ == 0) rows_per_core_ = neurons_per_core_;
    if (cols_per_core_ == 0) cols_per_core_ = neurons_per_core_;
    if (rows_per_core_ == 0 || cols_per_core_ == 0) {
        output_->fatal(CALL_INFO, -1, "❌ 无效的行/列配置: rows=%u cols=%u\n", rows_per_core_, cols_per_core_);
    }
    // 规范化chunk大小
    const uint32_t kMinChunk = 16;
    const uint32_t kMaxChunk = 1024 * 1024; // 1MiB保护
    if (chunk_size_bytes_ < kMinChunk) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ chunk_size_bytes过小(%u)，提升到%uB\n", chunk_size_bytes_, kMinChunk);
        chunk_size_bytes_ = kMinChunk;
    } else if (chunk_size_bytes_ > kMaxChunk) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ chunk_size_bytes过大(%u)，降低到%uB\n", chunk_size_bytes_, kMaxChunk);
        chunk_size_bytes_ = kMaxChunk;
    }
    // stride与期望长度比对（仅提示，不阻断）
    uint64_t expect_dense_bytes = static_cast<uint64_t>(rows_per_core_) * static_cast<uint64_t>(cols_per_core_) * sizeof(float);
    if (!raw_mode_ && per_core_stride_ != 0 && per_core_stride_ < expect_dense_bytes) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ per_core_stride(%" PRIu64 ") 小于期望的dense大小(%" PRIu64 ")，仍按实际长度写入\n", per_core_stride_, expect_dense_bytes);
    }
}


void WeightLoader::loadFileOnce() {
    // 优先从 single_file 或 file_template 载入；退化到 weight_file（旧参数）或填充值
    bool ok = false;
    if (!single_file_.empty()) {
        ok = loadSingleFileAllCores(single_file_, weight_format_);
    } else if (per_core_files_ && !file_template_.empty()) {
        ok = loadPerCoreFiles(file_template_, weight_format_);
    } else if (!weight_file_.empty()) {
        // 旧参数兼容：将其作为 single_file 使用
        ok = loadSingleFileAllCores(weight_file_, weight_format_);
    }
    if (!ok) {
        if (raw_mode_) {
            output_->fatal(CALL_INFO, -1, "❌ WeightLoader(raw) 未找到输入文件, 无法回退填充\n");
        } else if (bcsr_enable_) {
            output_->verbose(CALL_INFO, 1, 0, "⚙️ 生成BCSR填充权重: br=%u bc=%u pattern=%s\n", bcsr_br_, bcsr_bc_, bcsr_pattern_.c_str());
            issueWritesBCSRFill(fill_value_);
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 未提供可用权重文件，回退为填充值 %.3f\n", fill_value_);
            issueWritesFill(fill_value_);
        }
    }
    loaded_ = true;
}

void WeightLoader::issueWritesFill(float value) {
    if (!memory_) return;
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    const uint32_t chunk = std::max<uint32_t>(16, chunk_size_bytes_); // 下限保护

    uint64_t total_writes = 0;
    for (int core = 0; core < num_cores_; ++core) {
        const uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
        // 构造一行的连续字节缓冲
        std::vector<float> row_vals(C, value);
        const uint8_t* row_bytes = reinterpret_cast<const uint8_t*>(row_vals.data());
        const size_t row_bytes_len = static_cast<size_t>(C) * sizeof(float);
        for (uint32_t row = 0; row < R; ++row) {
            const uint64_t row_base_addr = base + static_cast<uint64_t>(row) * static_cast<uint64_t>(C) * sizeof(float);
            // 分块按 cacheline/chunk 写入
            size_t off = 0;
            while (off < row_bytes_len) {
                size_t len = std::min<size_t>(chunk, row_bytes_len - off);
                std::vector<uint8_t> data(row_bytes + off, row_bytes + off + len);
                sendWrite_(row_base_addr + off, data, /*timed*/false);
                off += len;
                total_writes++;
            }
        }
        output_->verbose(CALL_INFO, 2, 0, "   核心%d: base=%" PRIu64 " 行写合并 %u 行，行字节=%zu，chunk=%uB\n",
                          core, base, R, row_bytes_len, chunk);
    }
    output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader发出写请求数(合并后)=%" PRIu64 "\n", total_writes);
}

void WeightLoader::issueWritesBCSRFill(float value) {
    // 仅支持填充模式下的合成BCSR（diag 模式或 full 模式）
    if (!memory_) return;
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    uint32_t br = (bcsr_br_ ? bcsr_br_ : 16);
    uint32_t bc = (bcsr_bc_ ? bcsr_bc_ : 16);
    if (R == 0 || C == 0 || br == 0 || bc == 0) return;
    uint32_t nBR = (R + br - 1) / br;
    uint32_t nBC = (C + bc - 1) / bc;
    // 仅支持整除情形：否则截断到下取整块
    nBR = R / br;
    nBC = C / bc;
    if (nBR == 0 || nBC == 0) return;

    // pattern: diag -> 每个块行1个块在对角；full -> 每个块行包含全部块列
    auto blocks_in_row = [&](uint32_t row){
        if (bcsr_pattern_ == "full") return nBC;
        return (row < nBC) ? 1u : 0u; // diag
    };
    uint32_t total_blocks = 0;
    for (uint32_t r = 0; r < nBR; ++r) total_blocks += blocks_in_row(r);

    // Compute offsets
    size_t rowptr_bytes = (size_t)(nBR + 1) * sizeof(uint32_t);
    size_t colidx_bytes = (size_t)total_blocks * (size_t)bcsr_idx_bytes_;
    size_t block_bytes = (size_t)br * (size_t)bc * (size_t)bcsr_val_bytes_;
    uint64_t rp_off = 0;
    uint64_t ci_off = rp_off + rowptr_bytes;
    uint64_t bd_off = ci_off + colidx_bytes;

    // 最小stride提示
    uint64_t min_stride = bd_off + (uint64_t)total_blocks * block_bytes;
    if (per_core_stride_ != 0 && per_core_stride_ < min_stride) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ per_core_stride(%" PRIu64 ") 小于BCSR最小需求(%" PRIu64 ")，仍按实际长度写入\n",
                         per_core_stride_, min_stride);
    }

    // 准备对角colidx与rowptr
    std::vector<uint32_t> rowptr(nBR + 1, 0);
    std::vector<uint32_t> colidx;
    colidx.reserve(total_blocks);
    uint32_t acc = 0;
    for (uint32_t r = 0; r < nBR; ++r) {
        rowptr[r] = acc;
        uint32_t k = blocks_in_row(r);
        if (k == 1) {
            uint32_t c = (bcsr_pattern_ == "full") ? 0 : r; // diag: same; full: will be filled later
            colidx.push_back(c);
        } else if (k > 1) {
            for (uint32_t c = 0; c < nBC; ++c) colidx.push_back(c);
        }
        acc += k;
    }
    rowptr[nBR] = acc;

    // 写入每个核心的数据
    uint64_t total_writes = 0;
    for (int core = 0; core < num_cores_; ++core) {
        uint64_t base = base_addr_start_ + (uint64_t)core * per_core_stride_;
        output_->verbose(CALL_INFO, 1, 0,
            "[WL-addr] core=%d base=0x%llx rowptr_off=0x%llx colidx_off=0x%llx blockdata_off=0x%llx (stride=%" PRIu64 ")\n",
            core, (unsigned long long)base,
            (unsigned long long)(base + rp_off),
            (unsigned long long)(base + ci_off),
            (unsigned long long)(base + bd_off),
            per_core_stride_);
        // rowptr (uint32)
    const uint8_t* rb = reinterpret_cast<const uint8_t*>(rowptr.data());
        size_t off = 0; const uint32_t chunk = std::max<uint32_t>(16, chunk_size_bytes_);
        while (off < rowptr_bytes) {
            size_t len = std::min<size_t>(chunk, rowptr_bytes - off);
            std::vector<uint8_t> buf(rb + off, rb + off + len);
            if (verbose_ > 0 && per_core_stride_ > 0) {
                output_->verbose(CALL_INFO, 1, 0,
                    "[WL-write] core=%d ROWPTR addr=0x%llx len=%zu\n",
                    core, (unsigned long long)(base + rp_off + off), len);
            }
            sendWrite_(base + rp_off + off, buf, /*timed*/false);
            off += len; total_writes++;
        }
        // colidx (uint16/uint32)
        std::vector<uint8_t> ci;
        ci.resize(colidx_bytes);
        if (bcsr_idx_bytes_ == 2) {
            for (size_t i = 0; i < colidx.size(); ++i) {
                uint16_t v = (uint16_t)colidx[i];
                ci[i*2+0] = (uint8_t)(v & 0xFF);
                ci[i*2+1] = (uint8_t)((v >> 8) & 0xFF);
            }
        } else {
            for (size_t i = 0; i < colidx.size(); ++i) {
                uint32_t v = colidx[i];
                std::memcpy(&ci[i*4], &v, 4);
            }
        }
        off = 0;
        while (off < colidx_bytes) {
            size_t len = std::min<size_t>(chunk, colidx_bytes - off);
            std::vector<uint8_t> buf(ci.begin() + off, ci.begin() + off + len);
            if (verbose_ > 0 && per_core_stride_ > 0) {
                output_->verbose(CALL_INFO, 1, 0,
                    "[WL-write] core=%d COLIDX addr=0x%llx len=%zu\n",
                    core, (unsigned long long)(base + ci_off + off), len);
            }
            sendWrite_(base + ci_off + off, buf, /*timed*/false);
            off += len; total_writes++;
        }
        // blockdata (float/4B)
        std::vector<float> blk(br * bc, value);
        std::vector<uint8_t> blkbytes(block_bytes);
        if (bcsr_val_bytes_ == 4) {
            std::memcpy(blkbytes.data(), blk.data(), block_bytes);
        } else {
            // 仅支持FP32；其他情况填零
            std::fill(blkbytes.begin(), blkbytes.end(), 0);
        }
        for (uint32_t r = 0; r < nBR; ++r) {
            uint32_t k = blocks_in_row(r);
            if (k == 0) continue;
            for (uint32_t t = 0; t < k; ++t) {
                uint32_t idx_in_row = (k==1) ? 0 : t;
                // global block index = rowptr[r] + idx_in_row
                uint32_t gb = rowptr[r] + idx_in_row;
                uint64_t addr = base + bd_off + (uint64_t)gb * block_bytes;
                // write one block in chunks
                size_t offb = 0;
                while (offb < block_bytes) {
                    size_t len = std::min<size_t>(chunk, block_bytes - offb);
                    std::vector<uint8_t> buf(blkbytes.begin() + offb, blkbytes.begin() + offb + len);
                    if (verbose_ > 0 && per_core_stride_ > 0) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "[WL-write] core=%d BLOCK addr=0x%llx len=%zu row=%u blk_index=%u\n",
                            core, (unsigned long long)(addr + offb), len, r, gb);
                    }
                    sendWrite_(addr + offb, buf, /*timed*/false);
                    offb += len; total_writes++;
                }
            }
        }
    }
    output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader(BCSR)发出写请求数(合并后)=%" PRIu64 "\n", total_writes);
}

bool WeightLoader::readFileAllFloats(const std::string& path, const std::string& fmt, std::vector<float>& out) {
    out.clear();
    if (fmt == "bin") {
        std::ifstream fin(path, std::ios::binary);
        if (!fin.good()) return false;
        fin.seekg(0, std::ios::end);
        std::streamsize size = fin.tellg();
        fin.seekg(0, std::ios::beg);
        if (size <= 0 || size % sizeof(float) != 0) return false;
        size_t count = static_cast<size_t>(size / sizeof(float));
        out.resize(count);
        fin.read(reinterpret_cast<char*>(out.data()), size);
        return fin.good();
    } else {
        std::ifstream fin(path);
        if (!fin.good()) return false;
        std::string tok;
        while (fin >> tok) {
            try {
                out.push_back(std::stof(tok));
            } catch (...) {
                // 忽略格式不合法的token
            }
        }
        return !out.empty();
    }
}

bool WeightLoader::readFileRaw(const std::string& path, std::vector<uint8_t>& out) {
    out.clear();
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) return false;
    fin.seekg(0, std::ios::end);
    std::streamsize size = fin.tellg();
    if (size <= 0) {
        return false;
    }
    fin.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    fin.read(reinterpret_cast<char*>(out.data()), size);
    if (!fin && !fin.eof()) {
        return false;
    }
    std::streamsize read_bytes = fin.gcount();
    return read_bytes == size;
}

void WeightLoader::issueWritesRaw(int core, const std::vector<uint8_t>& data, bool timed) {
    if (!memory_) return;
    if (data.empty()) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 核心%d Raw写入数据为空，跳过\n", core);
        return;
    }
    uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
    if (verbose_ > 0) {
        output_->verbose(CALL_INFO, 1, 0,
            "[WL-addr] core=%d base=0x%llx stride=%" PRIu64 " bytes=0x%zx\n",
            core,
            (unsigned long long)base,
            per_core_stride_,
            data.size());
    }
    if (per_core_stride_ != 0 && data.size() > per_core_stride_) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d Raw数据长度(%zu)超过per_core_stride=%" PRIu64 "，仍按实际长度写入\n",
                         core, data.size(), per_core_stride_);
    }
        const uint32_t chunk = std::max<uint32_t>(16, chunk_size_bytes_);
    size_t off = 0;
    uint64_t total_writes = 0;
    while (off < data.size()) {
        size_t len = std::min<size_t>(chunk, data.size() - off);
        std::vector<uint8_t> payload(data.begin() + off, data.begin() + off + len);
        sendWrite_(base + off, payload, timed);
        off += len;
        total_writes++;
    }
    output_->verbose(CALL_INFO, 2, 0, "🧾 核心%d Raw写入: bytes=%zu, writes=%" PRIu64 "\n", core, data.size(), total_writes);
}

void WeightLoader::driveTimedJobs_() {
    if (!timed_seed_enable_) return;
    if (timed_raw_jobs_.empty()) {
        if (pending_writes_ == 0) {
            publishLoaderDone_();
        }
        return;
    }
    size_t iterations = timed_raw_jobs_.size();
    while (iterations-- > 0) {
        if (pending_writes_ >= timed_write_window_) break;
        auto job = timed_raw_jobs_.front();
        timed_raw_jobs_.pop_front();
        if (!job || job->offset >= job->data.size()) {
            if (job) std::vector<uint8_t>().swap(job->data);
            continue;
        }
        size_t remain = job->data.size() - job->offset;
        size_t len = std::min<size_t>(chunk_size_bytes_, remain);
        std::vector<uint8_t> payload(job->data.begin() + job->offset, job->data.begin() + job->offset + len);
        sendWrite_(job->base + job->offset, payload, /*timed*/true);
        job->offset += len;
        if (job->offset < job->data.size()) {
            timed_raw_jobs_.push_back(job);
        } else {
            std::vector<uint8_t>().swap(job->data);
        }
    }
    if (timed_raw_jobs_.empty() && pending_writes_ == 0) {
        publishLoaderDone_();
    }
}

void WeightLoader::issueWritesForCoreFloatsImpl(int core, const std::vector<float>& wbuf, bool timed) {
    if (!memory_) return;
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    const uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
    const size_t expected = static_cast<size_t>(R) * static_cast<size_t>(C);
    if (validate_length_ && wbuf.size() < expected) {
        output_->verbose(CALL_INFO, 1, 0, timed ? "⚠️ 运行时核心%d权重长度不足(%zu<%zu)，用fill_value补齐\n"
                                                : "⚠️ 核心%d权重长度不足(%zu<%zu)，用fill_value补齐\n",
                         core, wbuf.size(), expected);
    }
    const uint32_t chunk = std::max<uint32_t>(16, chunk_size_bytes_);
    // Row-major packing per row, then chunked writes
    for (uint32_t row = 0; row < R; ++row) {
        std::vector<float> row_vals(C, fill_value_);
        for (uint32_t col = 0; col < C; ++col) {
            size_t idx_rm = static_cast<size_t>(row) * static_cast<size_t>(C) + col;
            size_t idx_cm = static_cast<size_t>(col) * static_cast<size_t>(R) + row;
            float val = fill_value_;
            if (row_major_) {
                if (idx_rm < wbuf.size()) val = wbuf[idx_rm];
            } else {
                if (idx_cm < wbuf.size()) val = wbuf[idx_cm];
            }
            row_vals[col] = val;
        }
        const uint8_t* row_bytes = reinterpret_cast<const uint8_t*>(row_vals.data());
        const size_t row_bytes_len = static_cast<size_t>(C) * sizeof(float);
        const uint64_t row_base_addr = base + static_cast<uint64_t>(row) * static_cast<uint64_t>(C) * sizeof(float);
        size_t off = 0;
        while (off < row_bytes_len) {
            size_t len = std::min<size_t>(chunk, row_bytes_len - off);
            std::vector<uint8_t> data(row_bytes + off, row_bytes + off + len);
            sendWrite_(row_base_addr + off, data, timed);
            off += len;
        }
    }
}

void WeightLoader::issueWritesForCoreFloats(int core, const std::vector<float>& wbuf) {
    issueWritesForCoreFloatsImpl(core, wbuf, /*timed=*/false);
}

bool WeightLoader::loadSingleFileAllCores(const std::string& path, const std::string& fmt) {
    if (raw_mode_) {
        std::vector<uint8_t> raw;
        if (!readFileRaw(path, raw)) return false;
        if (num_cores_ <= 1) {
            issueWritesRaw(0, raw, false);
        } else {
            uint64_t stride = per_core_stride_ ? per_core_stride_ : raw.size();
            size_t offset = 0;
            for (int core = 0; core < num_cores_; ++core) {
                size_t remain = (offset < raw.size()) ? (raw.size() - offset) : 0;
                size_t len = std::min<size_t>(remain, static_cast<size_t>(stride));
                std::vector<uint8_t> slice;
                if (len > 0) {
                    slice.assign(raw.begin() + offset, raw.begin() + offset + len);
                }
                issueWritesRaw(core, slice, false);
                offset += len;
            }
            if (offset < raw.size()) {
                output_->verbose(CALL_INFO, 1, 0,
                    "⚠️ Raw单文件剩余%zu字节未分配给核心，已忽略\n", raw.size() - offset);
            }
        }
        output_->verbose(CALL_INFO, 1, 0, "✅ 原始权重加载完成: %s\n", path.c_str());
        return true;
    }
    std::vector<float> all;
    if (!readFileAllFloats(path, fmt, all)) return false;
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    const size_t per_core = static_cast<size_t>(R) * static_cast<size_t>(C);
    size_t offset = static_cast<size_t>(std::max(0, file_core_offset_)) * per_core;
    for (int core = 0; core < num_cores_; ++core) {
        size_t remain = (offset < all.size()) ? (all.size() - offset) : 0;
        std::vector<float> slice;
        if (remain >= per_core) {
            slice.assign(all.begin() + offset, all.begin() + offset + per_core);
        } else {
            slice.assign(all.begin() + offset, all.end());
        }
        issueWritesForCoreFloats(core, slice);
        offset += per_core;
    }
    output_->verbose(CALL_INFO, 1, 0, "✅ 单文件加载完成: %s\n", path.c_str());
    return true;
}

bool WeightLoader::loadPerCoreFiles(const std::string& tmpl, const std::string& fmt) {
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    const size_t per_core = static_cast<size_t>(R) * static_cast<size_t>(C);
    for (int core = 0; core < num_cores_; ++core) {
        // 支持多种模板格式：{core}、{core:02d}等
        std::string path = tmpl;
        
        // 处理 {core:02d} 格式
        size_t pos = path.find("{core:02d}");
        if (pos != std::string::npos) {
            char formatted[16];
            std::snprintf(formatted, sizeof(formatted), "%02d", core);
            path.replace(pos, 10, formatted);
        } else {
            // 处理简单的 {core} 格式
            pos = path.find("{core}");
            if (pos != std::string::npos) {
                path.replace(pos, 6, std::to_string(core));
            }
        }
        if (raw_mode_) {
            std::vector<uint8_t> raw;
            if (!readFileRaw(path, raw)) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 未找到核心%d原始文件 %s ，写入0字节\n", core, path.c_str());
                raw.clear();
            }
            issueWritesRaw(core, raw, false);
        } else {
            std::vector<float> buf;
            if (!readFileAllFloats(path, fmt, buf)) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 未找到核心%d文件 %s ，使用fill_value填充\n", core, path.c_str());
                buf.clear();
            }
            if (validate_length_ && buf.size() < per_core) {
                output_->verbose(CALL_INFO, 2, 0, "   核心%d文件长度不足(%zu<%zu)，补齐\n", core, buf.size(), per_core);
            }
            issueWritesForCoreFloats(core, buf);
        }
    }
    // output_->verbose(CALL_INFO, 1, 0, "✅ 按核心分文件加载完成: 模板 %s\n", tmpl.c_str());
    return true;
}

void WeightLoader::loadFileOnceRuntime() {
    // 运行时版本的loadFileOnce，使用时钟驱动的内存写入
    bool ok = false;
    if (!single_file_.empty()) {
        ok = loadSingleFileAllCores(single_file_, weight_format_);
    } else if (per_core_files_ && !file_template_.empty()) {
        ok = loadPerCoreFilesRuntime(file_template_, weight_format_);
    } else if (!weight_file_.empty()) {
        ok = loadSingleFileAllCores(weight_file_, weight_format_);
    }
    if (!ok) {
        if (raw_mode_) {
            output_->fatal(CALL_INFO, -1, "❌ WeightLoader(raw) 运行时未找到输入文件\n");
        } else {
            // 运行时回退：按 fill_value_ 进行逐行写入（timed），确保内存可读到填充值
            const uint32_t R = rows_per_core_;
            const uint32_t C = cols_per_core_;
            std::vector<float> row_vals(C, fill_value_);
            for (int core = 0; core < num_cores_; ++core) {
                // 构造整核缓冲（R×C）并调用 timed 写入
                std::vector<float> buf;
                buf.reserve(static_cast<size_t>(R) * static_cast<size_t>(C));
                for (uint32_t r = 0; r < R; ++r) {
                    buf.insert(buf.end(), row_vals.begin(), row_vals.end());
                }
                issueWritesForCoreFloatsRuntime(core, buf);
            }
            output_->verbose(CALL_INFO, 1, 0, "⚙️ 运行时回退：使用填充值 %.3f 写入权重 (timed)\n", fill_value_);
        }
    }
    loaded_ = true;
}

void WeightLoader::publishLoaderDone_() {
    if (!loader_done_shared_initialized_ || loader_done_published_) return;
    // 诊断打点：发布 Loader 完成标志，便于与核心侧 ensureLoaderReady_ 对齐
    if (output_) {
        output_->verbose(CALL_INFO, 1, 0,
            "[WL-done] publish loader_done_key=%s (rows=%u cols=%u num_cores=%d base=0x%llx stride=%" PRIu64 ")\n",
            loader_done_key_.c_str(), rows_per_core_, cols_per_core_, num_cores_,
            (unsigned long long)base_addr_start_, per_core_stride_);
    }
    loader_done_shared_.write(0, 1);
    loader_done_shared_.publish();
    loader_done_published_ = true;
}

bool WeightLoader::loadPerCoreFilesRuntime(const std::string& tmpl, const std::string& fmt) {
    const uint32_t R = rows_per_core_;
    const uint32_t C = cols_per_core_;
    const size_t per_core = static_cast<size_t>(R) * static_cast<size_t>(C);
    for (int core = 0; core < num_cores_; ++core) {
        std::string path = tmpl;
        
        size_t pos = path.find("{core:02d}");
        if (pos != std::string::npos) {
            char formatted[16];
            std::snprintf(formatted, sizeof(formatted), "%02d", core);
            path.replace(pos, 10, formatted);
        } else {
            pos = path.find("{core}");
            if (pos != std::string::npos) {
                path.replace(pos, 6, std::to_string(core));
            }
        }
        
        if (raw_mode_) {
            std::vector<uint8_t> raw;
            if (!readFileRaw(path, raw)) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 运行时未找到核心%d原始文件 %s\n", core, path.c_str());
                continue;
            }
            auto job = std::make_shared<TimedRawJob>();
            job->core = core;
            job->base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
            job->data = std::move(raw);
            job->offset = 0;
            timed_raw_jobs_.push_back(job);
            driveTimedJobs_();
        } else {
            std::vector<float> buf;
            if (!readFileAllFloats(path, fmt, buf)) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 运行时未找到核心%d文件 %s\n", core, path.c_str());
                buf.clear();
            }
            if (validate_length_ && buf.size() < per_core) {
                output_->verbose(CALL_INFO, 2, 0, "   运行时核心%d文件长度不足(%zu<%zu)\n", core, buf.size(), per_core);
            }
            issueWritesForCoreFloatsRuntime(core, buf);
        }
    }
    // output_->verbose(CALL_INFO, 1, 0, "✅ 运行时按核心分文件加载完成: 模板 %s\n", tmpl.c_str());
    return true;
}

void WeightLoader::issueWritesForCoreFloatsRuntime(int core, const std::vector<float>& wbuf) {
    issueWritesForCoreFloatsImpl(core, wbuf, /*timed=*/true);
}

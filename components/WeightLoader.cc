// -*- c++ -*-

#include <sst/core/sst_config.h>
#include "WeightLoader.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <inttypes.h>
#include <cctype>
#include <iterator>

using namespace SST;
using namespace SST::SnnDL;

std::string WeightLoader::hexDump_(const std::vector<uint8_t>& buf, size_t max_bytes) {
    const size_t n = std::min(max_bytes, buf.size());
    std::string s;
    s.reserve(n * 3);
    for (size_t i = 0; i < n; ++i) {
        char t[8];
        std::snprintf(t, sizeof(t), "%02x", buf[i]);
        if (i) s.push_back(' ');
        s += t;
    }
    return s;
}

bool WeightLoader::readFileSlice_(const std::string& path, uint64_t off, size_t len, std::vector<uint8_t>& out) {
    out.clear();
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) return false;
    fin.seekg(0, std::ios::end);
    std::streamoff sz = fin.tellg();
    if (sz <= 0) return false;
    if (off >= static_cast<uint64_t>(sz)) return false;
    const size_t can = static_cast<size_t>(std::min<uint64_t>(static_cast<uint64_t>(sz) - off, static_cast<uint64_t>(len)));
    fin.seekg(static_cast<std::streamoff>(off), std::ios::beg);
    out.resize(can);
    fin.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(can));
    return fin.good() || fin.eof();
}

bool WeightLoader::parseMetaU64_(const std::string& meta_text, const char* key, uint64_t& out) {
    auto pos = meta_text.find(key);
    if (pos == std::string::npos) return false;
    pos = meta_text.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < meta_text.size() && (std::isspace(static_cast<unsigned char>(meta_text[pos])) || meta_text[pos] == '"')) ++pos;
    size_t end = pos;
    while (end < meta_text.size() && (std::isdigit(static_cast<unsigned char>(meta_text[end])) || meta_text[end] == 'x' || meta_text[end] == 'X')) ++end;
    if (end <= pos) return false;
    out = std::strtoull(meta_text.substr(pos, end - pos).c_str(), nullptr, 0);
    return true;
}

bool WeightLoader::parseMetaU32_(const std::string& meta_text, const char* key, uint32_t& out) {
    uint64_t tmp = 0;
    if (!parseMetaU64_(meta_text, key, tmp)) return false;
    out = static_cast<uint32_t>(tmp);
    return true;
}

std::string WeightLoader::resolveCorePath_(int core, bool meta) const {
    if (!per_core_files_ || file_template_.empty()) return "";
    std::string p = file_template_;
    {
        auto pos = p.find("{core:02d}");
        if (pos != std::string::npos) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%02d", core);
            p.replace(pos, 10, buf);
        } else if ((pos = p.find("{core}")) != std::string::npos) {
            p.replace(pos, 6, std::to_string(core));
        }
    }
    if (meta) p += ".meta.json";
    return p;
}

void WeightLoader::issueVerifyReadbacks_() {
    if (!verify_readback_enable_ || verify_readback_issued_ || verify_readback_done_) return;
    if (!memory_) return;
    if (!raw_mode_ || !bcsr_enable_) return;

    const int core = verify_readback_core_;
    if (core < 0 || core >= num_cores_) return;
    const std::string bin_path = resolveCorePath_(core, /*meta=*/false);
    const std::string meta_path = resolveCorePath_(core, /*meta=*/true);
    if (bin_path.empty() || meta_path.empty()) return;

    std::ifstream mf(meta_path);
    if (!mf.good()) return;
    std::string mt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    uint64_t colidx_off = 0;
    uint32_t idx_bytes = 0;
    if (!parseMetaU64_(mt, "\"colidx_offset\"", colidx_off)) return;
    if (!parseMetaU32_(mt, "\"idx_bytes\"", idx_bytes)) idx_bytes = bcsr_idx_bytes_;
    if (idx_bytes == 0) idx_bytes = bcsr_idx_bytes_;
    const uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;

    struct Sample { const char* tag; uint64_t file_off; uint64_t mem_addr; };
    const uint64_t s1 = colidx_off;
    const uint64_t s2 = colidx_off + static_cast<uint64_t>(verify_colidx_start_index_) * static_cast<uint64_t>(idx_bytes);
    const Sample samples[] = {
        {"colidx@0", s1, base + s1},
        {"colidx@start", s2, base + s2},
    };

    for (const auto& s : samples) {
        std::vector<uint8_t> expect;
        if (!readFileSlice_(bin_path, s.file_off, verify_readback_bytes_, expect)) continue;
        auto* r = new SST::Interfaces::StandardMem::Read(s.mem_addr, expect.size());
        const auto id = r->getID();
        VerifyPending vp;
        vp.addr = s.mem_addr;
        vp.tag = s.tag;
        vp.expect = std::move(expect);
        verify_pending_[id] = std::move(vp);
        memory_->sendUntimedData(r);
        if (output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-verify-issue] core=%d tag=%s addr=0x%llx bytes=%zu file_off=0x%llx\n",
                core, s.tag,
                (unsigned long long)s.mem_addr,
                verify_pending_[id].expect.size(),
                (unsigned long long)s.file_off);
        }
    }
    verify_readback_issued_ = !verify_pending_.empty();
}

void WeightLoader::pollVerifyReadbacks_() {
    if (!verify_readback_enable_ || verify_readback_done_) return;
    if (!memory_) return;
    if (verify_pending_.empty()) {
        if (verify_readback_issued_) verify_readback_done_ = true;
        return;
    }
    // init/complete 阶段：需要主动 poll recvUntimedData()
    while (true) {
        auto* req = memory_->recvUntimedData();
        if (!req) break;
        auto* rr = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
        if (!rr) {
            delete req;
            continue;
        }
        auto it = verify_pending_.find(rr->getID());
        if (it == verify_pending_.end()) {
            delete req;
            continue;
        }
        const auto& expect = it->second.expect;
        const auto& got = rr->data;
        const size_t n = std::min(expect.size(), got.size());
        size_t mismatch = 0;
        for (size_t i = 0; i < n; ++i) {
            if (expect[i] != got[i]) mismatch++;
        }
        const bool size_ok = got.size() == expect.size();
        const bool ok = size_ok && (mismatch == 0);
        if (!ok) {
            verify_failed_ = true;
            if (strict_loader_done_ && output_) {
                output_->fatal(CALL_INFO, -1,
                    "❌ WeightLoader strict_loader_done: verify_readback 失败: tag=%s addr=0x%llx got=%zu expect=%zu mismatch=%zu。"
                    "若 got=0，可能 untimed ReadResp 不携带 data；请关闭 strict_loader_done 或改用其他验证手段。\n",
                    it->second.tag.c_str(),
                    (unsigned long long)it->second.addr,
                    got.size(), expect.size(), mismatch);
            }
        }
        if (output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-verify-resp] tag=%s addr=0x%llx got=%zu expect=%zu mismatch=%zu head_got=[%s] head_expect=[%s]\n",
                it->second.tag.c_str(),
                (unsigned long long)it->second.addr,
                got.size(), expect.size(), mismatch,
                hexDump_(got, 16).c_str(),
                hexDump_(expect, 16).c_str());
        }
        verify_pending_.erase(it);
        delete req;
    }
    if (verify_pending_.empty() && verify_readback_issued_) {
        verify_readback_done_ = true;
    }
}

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
    verify_readback_enable_ = params.find<int>("verify_readback_enable", 0) != 0;
    verify_readback_core_ = params.find<int>("verify_readback_core", 0);
    verify_readback_bytes_ = params.find<uint32_t>("verify_readback_bytes", 64);
    verify_colidx_start_index_ = params.find<uint32_t>("verify_colidx_start_index", 441);
    strict_loader_done_ = params.find<int>("strict_loader_done", 0) != 0;
    min_raw_bcsr_chunk_bytes_ = params.find<uint32_t>("min_raw_bcsr_chunk_bytes", 4096);
    if (min_raw_bcsr_chunk_bytes_ != 0 && min_raw_bcsr_chunk_bytes_ < 64) {
        min_raw_bcsr_chunk_bytes_ = 64;
    }
    // strict_loader_done 仅对 raw+BCSR 有意义；其他模式直接忽略，避免误伤旧用例。
    if (strict_loader_done_ && !(raw_mode_ && bcsr_enable_)) {
        strict_loader_done_ = false;
    }
    if (strict_loader_done_) {
        // strict 模式下必须启用写后读回校验；否则 loader_done 会在“写入不可见/丢写”时误放行，导致读回全0。
        verify_readback_enable_ = true;
    }
    diag_runtime_read_enable_ = params.find<int>("diag_runtime_read_enable", 0) != 0;
    diag_runtime_read_core_ = params.find<int>("diag_runtime_read_core", 0);
    diag_runtime_read_offset_ = params.find<uint64_t>("diag_runtime_read_offset", 0);
    diag_runtime_read_bytes_ = params.find<uint32_t>("diag_runtime_read_bytes", 64);

    output_ = new Output("WeightLoader[@p:@l]: ", verbose_, 0, Output::STDOUT);
    // 无条件记录构造基本配置，便于确认是否实际创建
    output_->verbose(CALL_INFO, 0, 0,
        "[WL-diag-init] verbose=%d weight_file=%s file_tmpl=%s single_file=%s base=0x%llx stride=0x%llx num_cores=%d rows=%u cols=%u fill=%.3f raw=%d bcsr=%d\n",
        verbose_, weight_file_.c_str(), file_template_.c_str(), single_file_.c_str(),
        (unsigned long long)base_addr_start_, (unsigned long long)per_core_stride_,
        num_cores_, rows_per_core_, cols_per_core_, fill_value_, raw_mode_ ? 1 : 0, bcsr_enable_ ? 1 : 0);

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

    // 运行期单点读回探针地址（仅诊断）：base(core)+offset
    if (diag_runtime_read_enable_) {
        const int c = diag_runtime_read_core_;
        if (c < 0 || c >= num_cores_) {
            output_->fatal(CALL_INFO, -1, "❌ diag_runtime_read_core=%d 超出范围 [0,%d)\n", c, num_cores_);
        }
        diag_runtime_read_addr_ =
            base_addr_start_ + static_cast<uint64_t>(c) * per_core_stride_ + diag_runtime_read_offset_;
        if (diag_runtime_read_bytes_ == 0) diag_runtime_read_bytes_ = 64;
    }
}

WeightLoader::~WeightLoader() {
    delete output_;
}

void WeightLoader::init(unsigned int phase) {
    // 将init相位转发给StandardMem
    if (memory_) memory_->init(phase);

    if (phase == 0) {
        if (output_ && verbose_ >= 2) {
            output_->verbose(CALL_INFO, 1, 0,
                "[WL-init] loader_done_key=%s timed=%d allow_cache=%d base=0x%llx stride=%" PRIu64 " cores=%d\n",
                loader_done_key_.c_str(),
                timed_seed_enable_ ? 1 : 0,
                timed_seed_allow_cache_ ? 1 : 0,
                (unsigned long long)base_addr_start_, per_core_stride_, num_cores_);
        }
    }
    if (phase == 1 && !loaded_) {
        // 关键约束：BCSR 权重数据规模很大（每核 MB 级），若在 timed 仿真阶段通过 memHierarchy
        // 逐 cacheline 拆分写入，会消耗远超 100us 的模拟时间并导致核心侧长期 loader-not-ready。
        // 因此：无论是否开启 timed seed，都在 init 阶段完成一次性 untimed 加载（不计入模拟时间）。
        runtime_load_needed_ = false;
        loadFileOnce();
        loaded_ = true;
        // 写入后在 init/complete 阶段发起读回校验（仅用于定位；注意：StandardMem 的 untimed ReadResp 可能不携带 data）。
        // strict_loader_done=1 时：必须校验通过后才发布 loader_done，避免“权重写入不可见→读回全0→发放归零”。
        issueVerifyReadbacks_();
        if (strict_loader_done_) {
            if (!verify_readback_issued_) {
                output_->fatal(CALL_INFO, -1,
                    "❌ WeightLoader strict_loader_done=1 但未能发起任何 verify_readback 请求。"
                    "请确认 per_core_files=1 且 file_template 指向可读的 *.bcsr.bin 文件（带 .meta.json），"
                    "或关闭 strict_loader_done。\n");
            }
        } else {
            publishLoaderDone_();
        }
    }
}

void WeightLoader::setup() {
    if (memory_) {
        memory_->setup();
    }
    // output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader setup完成\n");
    
    // 统一语义：权重必须在 init 阶段完成加载并发布 loader_done；
    // timed_seed_* 不再触发“运行期全量写入”（避免模拟时间被 WeightLoader 吞没）。
    if (!loaded_) {
        output_->fatal(CALL_INFO, -1, "❌ WeightLoader setup 阶段检测到未完成的权重加载（expected in init）。");
    }
    if (strict_loader_done_) {
        if (verify_failed_) {
            output_->fatal(CALL_INFO, -1, "❌ WeightLoader strict_loader_done: 写后读回校验失败，拒绝发布 loader_done。\n");
        }
        if (!verify_readback_done_) {
            output_->fatal(CALL_INFO, -1,
                "❌ WeightLoader strict_loader_done: 写后读回校验未完成（可能 untimed ReadResp 不可用/未返回 data），拒绝发布 loader_done。\n");
        }
        publishLoaderDone_();
    }
    runtime_load_needed_ = false;

    // 调试：启用运行期单点读回探针（用于验证 timed read 在当前拓扑下读取到的字节）
    if (diag_runtime_read_enable_ && !clock_registered_) {
        registerClock("1GHz", new SST::Clock::Handler2<WeightLoader, &WeightLoader::onClockTick>(this));
        clock_registered_ = true;
    }
}

void WeightLoader::finish() {
    if (memory_) {
        memory_->finish();
    }
    // output_->verbose(CALL_INFO, 1, 0, "🏁 WeightLoader 完成\n");
}

bool WeightLoader::onClockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;
    if (diag_runtime_read_enable_ && !diag_runtime_read_issued_) {
        if (!memory_) return false;
        auto* r = new SST::Interfaces::StandardMem::Read(diag_runtime_read_addr_, diag_runtime_read_bytes_);
        diag_runtime_read_id_ = r->getID();
        diag_runtime_read_issued_ = true;
        memory_->send(r);
        if (output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-diag-timed-read-issue] core=%d addr=0x%llx bytes=%u\n",
                diag_runtime_read_core_,
                (unsigned long long)diag_runtime_read_addr_,
                diag_runtime_read_bytes_);
        }
    }
    return false;
}

void WeightLoader::complete(unsigned int phase) {
    // 将 StandardMem 的 complete 透传
    if (memory_) memory_->complete(phase);
    // init/complete 阶段：poll 读回校验响应（仅调试）
    pollVerifyReadbacks_();
    if (phase == 0 && !loaded_ && !runtime_load_needed_) {
        output_->verbose(CALL_INFO, 1, 0,
            "[WL-complete] phase=%u notice: loaded=%d runtime_load_needed=%d", phase,
            loaded_ ? 1 : 0, runtime_load_needed_ ? 1 : 0);
    }
}
void WeightLoader::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
    if (!req) return;

    // 运行期单点读回探针：打印 timed ReadResp 的首字节（对齐文件预期）
    if (diag_runtime_read_enable_ && diag_runtime_read_issued_ && req->getID() == diag_runtime_read_id_) {
        auto* rr = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
        if (!rr) {
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-diag-timed-read-resp] id=%" PRIu64 " (non-ReadResp)\n",
                req->getID());
            delete req;
            return;
        }
        output_->verbose(CALL_INFO, 0, 0,
            "[WL-diag-timed-read-resp] id=%" PRIu64 " addr=0x%llx got=%zu head=[%s]\n",
            rr->getID(),
            (unsigned long long)rr->pAddr,
            rr->data.size(),
            hexDump_(rr->data, 16).c_str());
        delete req;
        return;
    }

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
    // Guard: raw+BCSR 大文件禁止使用过小 chunk 做 untimed bulk writes。
    // 否则将产生海量 sendUntimedData(Write)，容易出现写入不可见/丢写，最终表现为“读回全0→发放归零”。
    if (!timed && raw_mode_ && bcsr_enable_ && min_raw_bcsr_chunk_bytes_ != 0) {
        const size_t kLargeFileBytes = 1024 * 1024; // 1MiB
        if (data.size() >= kLargeFileBytes && chunk_size_bytes_ < min_raw_bcsr_chunk_bytes_) {
            const size_t chunk = std::max<size_t>(16, static_cast<size_t>(chunk_size_bytes_));
            const size_t est_writes_per_core = (data.size() + chunk - 1) / chunk;
            const size_t est_writes_total = est_writes_per_core * static_cast<size_t>(std::max(1, num_cores_));
            output_->fatal(CALL_INFO, -1,
                "❌ WeightLoader raw+BCSR: chunk_size_bytes=%u 太小（min=%u），文件=0x%zx，将导致 ~%zu writes/core (~%zu writes/PE)。"
                "请提高 loader_chunk_bytes（推荐 65536；至少 4096），或将 min_raw_bcsr_chunk_bytes=0 关闭此保护。\n",
                chunk_size_bytes_,
                min_raw_bcsr_chunk_bytes_,
                data.size(),
                est_writes_per_core,
                est_writes_total);
        }
    }
    // 若配置了 per_core_stride，则要求其必须覆盖文件长度；否则权重地址映射会跨 core/PE 导致非确定性/读回垃圾值
    if (per_core_stride_ != 0 && data.size() > static_cast<size_t>(per_core_stride_)) {
        output_->fatal(CALL_INFO, -1,
            "❌ WeightLoader raw 文件长度(0x%zx)超过 per_core_stride(0x%llx)。"
            " 这会导致地址映射跨 core 溢出并产生非确定性。请增大 stride（建议取所有 PE/core 的 file_size 最大值并 64B 对齐）。\n",
            data.size(), (unsigned long long)per_core_stride_);
    }
    const size_t write_limit = data.size();
    // 诊断：尝试读取对应 meta 以得到 BCSR offset，便于对齐校验
    uint64_t diag_rowptr_off = 0, diag_colidx_off = 0, diag_blockdata_off = 0;
    bool diag_meta_ok = false;
    if (bcsr_enable_ && per_core_files_ && !file_template_.empty()) {
        std::string meta_path = file_template_;
        // 仅替换 {core}，保留 {pe}
        {
            auto p = meta_path.find("{core:02d}");
            if (p != std::string::npos) {
                char buf[8]; std::snprintf(buf, sizeof(buf), "%02d", core);
                meta_path.replace(p, 10, buf); // 含末尾右括号
            } else if ((p = meta_path.find("{core}")) != std::string::npos) {
                meta_path.replace(p, 6, std::to_string(core));
            }
        }
        // 将 .bin 替换为 .bin.meta.json
        if (meta_path.size() >= 4 && meta_path.rfind(".bin") == meta_path.size() - 4) {
            meta_path += ".meta.json";
        } else {
            meta_path += ".meta.json";
        }
        std::ifstream mf(meta_path);
        if (mf.good()) {
            std::string mt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
            auto parseU64 = [&](const char* key, uint64_t& out)->bool{
                auto pos = mt.find(key);
                if (pos == std::string::npos) return false;
                pos = mt.find(':', pos);
                if (pos == std::string::npos) return false;
                ++pos;
                while (pos < mt.size() && (std::isspace(static_cast<unsigned char>(mt[pos])) || mt[pos] == '\"')) ++pos;
                size_t end = pos;
                while (end < mt.size() && (std::isdigit(static_cast<unsigned char>(mt[end])) || mt[end]=='x' || mt[end]=='X')) ++end;
                if (end <= pos) return false;
                out = std::strtoull(mt.substr(pos, end-pos).c_str(), nullptr, 0);
                return true;
            };
            uint64_t tmp=0;
            if (parseU64("\"rowptr_offset\"", diag_rowptr_off)) diag_meta_ok = true;
            if (parseU64("\"colidx_offset\"", tmp)) { diag_colidx_off = tmp; diag_meta_ok = true; }
            if (parseU64("\"blockdata_offset\"", tmp)) { diag_blockdata_off = tmp; diag_meta_ok = true; }
        }
        if (!diag_meta_logged_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-diag-meta] core=%d meta_path=%s ok=%d rp=0x%llx ci=0x%llx bd=0x%llx\n",
                core, meta_path.c_str(), diag_meta_ok ? 1 : 0,
                (unsigned long long)diag_rowptr_off,
                (unsigned long long)diag_colidx_off,
                (unsigned long long)diag_blockdata_off);
            diag_meta_logged_ = true;
        }
    }
    uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
    // 无条件记录 Raw 写入的地址区间和首个非零字节
    uint8_t sample_nz = 0;
    for (auto b : data) { if (b != 0) { sample_nz = b; break; } }
    output_->verbose(CALL_INFO, 0, 0,
        "[WL-raw] core=%d base=0x%llx end=0x%llx stride=0x%llx bytes=0x%zx sample_nz=0x%02x timed=%d\n",
        core,
        (unsigned long long)base,
        (unsigned long long)(base + data.size()),
        (unsigned long long)per_core_stride_,
        data.size(),
        sample_nz,
        timed ? 1 : 0);
    if (per_core_stride_ != 0 && data.size() > per_core_stride_) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d Raw数据长度(%zu)超过per_core_stride=%" PRIu64 "，仍按实际长度写入\n",
                         core, data.size(), per_core_stride_);
    }
    const uint32_t chunk = std::max<uint32_t>(16, chunk_size_bytes_);
    size_t off = 0;
    uint64_t total_writes = 0;
    static int diag_overlap_count = 0;
    while (off < write_limit) {
        size_t len = std::min<size_t>(chunk, write_limit - off);
        std::vector<uint8_t> payload(data.begin() + off, data.begin() + off + len);
        // 诊断：无条件记录 core0 前几个 chunk（含 hex），并注明 meta offset（每实例独立计数）
        if (core == 0 && diag_chunk_count_ < 12) {
            auto dump_hex = [&](const std::vector<uint8_t>& buf)->std::string{
                const size_t dump = std::min<size_t>(buf.size(), 32);
                std::string s;
                for (size_t i=0;i<dump;++i){ char t[8]; std::snprintf(t,sizeof(t),"%02x", buf[i]); if(i) s.push_back(' '); s += t; }
                return s;
            };
            output_->output("WeightLoader[issueWritesRaw]: [WL-diag-chunk] core=%d chunk_addr=0x%llx len=%zu rowptr_off=0x%llx colidx_off=0x%llx blockdata_off=0x%llx hex=[%s]\n",
                core,
                (unsigned long long)(base + off), len,
                (unsigned long long)diag_rowptr_off,
                (unsigned long long)diag_colidx_off,
                (unsigned long long)diag_blockdata_off,
                dump_hex(payload).c_str());
            ++diag_chunk_count_;
        } else if (diag_meta_ok && diag_chunk_count_ < 32) {
            uint64_t addr_start = base + off;
            uint64_t addr_end = addr_start + len;
            auto dump_hex = [&](const std::vector<uint8_t>& buf)->std::string{
                const size_t dump = std::min<size_t>(buf.size(), 16);
                std::string s;
                for (size_t i=0;i<dump;++i){ char t[8]; std::snprintf(t,sizeof(t),"%02x", buf[i]); if(i) s.push_back(' '); s += t; }
                return s;
            };
            if (addr_start <= base + diag_colidx_off && addr_end > base + diag_colidx_off) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[WL-diag-chunk] core=%d chunk_addr=0x%llx len=%zu overlaps colidx_off=0x%llx rowptr_off=0x%llx blockdata_off=0x%llx hex=[%s]\n",
                    core,
                    (unsigned long long)addr_start, len,
                    (unsigned long long)diag_colidx_off,
                    (unsigned long long)diag_rowptr_off,
                    (unsigned long long)diag_blockdata_off,
                    dump_hex(payload).c_str());
                ++diag_chunk_count_;
            }
        }
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
        sendWrite_(job->base + job->offset, payload, /*timed=*/true);
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
    // 诊断：一次性记录写入地址区间与样本值，核对与 PE 基址/stride 是否一致
    static std::unordered_set<int> logged_cores;
    if (logged_cores.find(core) == logged_cores.end()) {
        float sample_nonzero = 0.0f;
        for (size_t i = 0; i < wbuf.size(); ++i) {
            if (wbuf[i] != 0.0f) { sample_nonzero = wbuf[i]; break; }
        }
        uint64_t end_addr = base + static_cast<uint64_t>(R) * static_cast<uint64_t>(C) * sizeof(float);
        output_->verbose(CALL_INFO, 0, 0,
            "[WL-diag] core=%d base=0x%llx end=0x%llx stride=0x%llx rows=%u cols=%u sample_nonzero=%.6f\n",
            core, (unsigned long long)base, (unsigned long long)end_addr,
            (unsigned long long)per_core_stride_, R, C, sample_nonzero);
        logged_cores.insert(core);
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
    output_->verbose(CALL_INFO, 0, 0, "[WL-diag] loadSingleFileAllCores path=%s fmt=%s raw=%d\n",
        path.c_str(), fmt.c_str(), raw_mode_ ? 1 : 0);
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
    output_->verbose(CALL_INFO, 0, 0, "[WL-diag] loadPerCoreFiles tmpl=%s fmt=%s raw=%d\n",
        tmpl.c_str(), fmt.c_str(), raw_mode_ ? 1 : 0);
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
            // 解析 meta 以便对齐诊断（仅限存在 .meta.json 时）
            uint64_t diag_rowptr_off = 0, diag_colidx_off = 0, diag_blockdata_off = 0, diag_blockids_off = 0;
            bool diag_meta_ok = false;
            std::string meta_path = path + ".meta.json";
            std::ifstream mf(meta_path);
            if (mf.good()) {
                std::string mt((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
                auto parseU64 = [&](const char* key, uint64_t& out)->bool{
                    auto pos = mt.find(key);
                    if (pos == std::string::npos) return false;
                    pos = mt.find(':', pos);
                    if (pos == std::string::npos) return false;
                    ++pos;
                    while (pos < mt.size() && (std::isspace(static_cast<unsigned char>(mt[pos])) || mt[pos] == '\"')) ++pos;
                    size_t end = pos;
                    while (end < mt.size() && (std::isdigit(static_cast<unsigned char>(mt[end])) || mt[end]=='x' || mt[end]=='X')) ++end;
                    if (end <= pos) return false;
                    out = std::strtoull(mt.substr(pos, end-pos).c_str(), nullptr, 0);
                    return true;
                };
                uint64_t tmp=0;
                if (parseU64("\"rowptr_offset\"", diag_rowptr_off)) diag_meta_ok = true;
                if (parseU64("\"colidx_offset\"", tmp)) { diag_colidx_off = tmp; diag_meta_ok = true; }
                if (parseU64("\"blockdata_offset\"", tmp)) { diag_blockdata_off = tmp; diag_meta_ok = true; }
                if (parseU64("\"blockids_offset\"", tmp)) { diag_blockids_off = tmp; diag_meta_ok = true; }
                output_->verbose(CALL_INFO, 0, 0,
                    "[WL-diag-meta-timed] core=%d meta=%s ok=%d rp=0x%llx ci=0x%llx bd=0x%llx ids=0x%llx\n",
                    core, meta_path.c_str(), diag_meta_ok ? 1 : 0,
                    (unsigned long long)diag_rowptr_off,
                    (unsigned long long)diag_colidx_off,
                    (unsigned long long)diag_blockdata_off,
                    (unsigned long long)diag_blockids_off);
            }
            // 诊断：记录队列化的写入区间与样本非零字节
            uint8_t sample_nz = 0;
            for (auto b : raw) { if (b != 0) { sample_nz = b; break; } }
            uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
            output_->verbose(CALL_INFO, 0, 0,
                "[WL-timed-enqueue] core=%d base=0x%llx end=0x%llx stride=0x%llx bytes=0x%zx sample_nz=0x%02x meta_ok=%d\n",
                core,
                (unsigned long long)base,
                (unsigned long long)(base + raw.size()),
                (unsigned long long)per_core_stride_,
                raw.size(),
                sample_nz,
                diag_meta_ok ? 1 : 0);
            auto job = std::make_shared<TimedRawJob>();
            job->core = core;
            job->base = base;
            job->data = std::move(raw);
            job->offset = 0;
            job->rowptr_off = diag_rowptr_off;
            job->colidx_off = diag_colidx_off;
            job->blockdata_off = diag_blockdata_off;
            job->blockids_off = diag_blockids_off;
            job->meta_ok = diag_meta_ok;
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

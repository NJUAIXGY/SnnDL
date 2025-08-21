// -*- c++ -*-

#include <sst/core/sst_config.h>
#include "WeightLoader.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

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
    fill_value_ = params.find<float>("fill_value", 0.5f);
    weight_format_ = params.find<std::string>("weight_format", "bin");
    per_core_files_ = params.find<int>("per_core_files", 0) != 0;
    file_template_ = params.find<std::string>("file_template", "");
    single_file_ = params.find<std::string>("single_file", "");
    row_major_ = params.find<int>("row_major", 1) != 0;
    chunk_size_bytes_ = params.find<uint32_t>("chunk_size_bytes", 64);
    validate_length_ = params.find<int>("validate_length", 1) != 0;
    file_core_offset_ = params.find<int>("file_core_offset", 0);
    timed_seed_enable_ = params.find<int>("timed_seed_enable", 1) != 0;
    timed_seed_count_ = params.find<uint32_t>("timed_seed_count", 1);

    output_ = new Output("WeightLoader[@p:@l]: ", verbose_, 0, Output::STDOUT);
    output_->verbose(CALL_INFO, 1, 0, "🔧 初始化WeightLoader\n");

    // 加载 StandardMem 子组件
    memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memory", ComponentInfo::SHARE_NONE,
        registerTimeBase("1ns"),
        new SST::Interfaces::StandardMem::Handler2<WeightLoader, &WeightLoader::handleMemoryResponse>(this));
    if (!memory_) {
        output_->fatal(CALL_INFO, -1, "❌ WeightLoader未配置StandardMem子组件\n");
    }
}

WeightLoader::~WeightLoader() {
    delete output_;
}

void WeightLoader::init(unsigned int phase) {
    // 将init相位转发给StandardMem
    if (memory_) memory_->init(phase);

    if (phase == 1 && !loaded_) {
        loadFileOnce();
    }
}

void WeightLoader::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader setup完成\n");
    
    if (!clock_registered_) {
        registerClock("1GHz", new SST::Clock::Handler<WeightLoader>(this, &WeightLoader::onClockTick));
        clock_registered_ = true;
    }
    
    // 在setup完成后的第一个时钟周期进行权重加载
    runtime_load_needed_ = true;
}

void WeightLoader::finish() {
    output_->verbose(CALL_INFO, 1, 0, "🏁 WeightLoader 完成\n");
}

bool WeightLoader::onClockTick(SST::Cycle_t cycle) {
    current_cycle_ = cycle;
    
    // 在运行时第一个周期进行权重加载
    if (runtime_load_needed_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔄 在运行时第一个周期进行权重加载\n");
        runtime_load_needed_ = false;
        loaded_ = false;  // 重置标志以允许重新加载
        loadFileOnceRuntime();  // 使用运行时加载函数
    }
    
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
            output_->verbose(CALL_INFO, 1, 0, "🎉 所有权重写入操作已完成！\n");
        }
    }
    
    delete req;
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
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 未提供可用权重文件，回退为填充值 %.3f\n", fill_value_);
        issueWritesFill(fill_value_);
    }
    loaded_ = true;
}

void WeightLoader::issueWritesFill(float value) {
    if (!memory_) return;
    const uint32_t N = neurons_per_core_;
    std::vector<uint8_t> data(sizeof(float));
    std::memcpy(data.data(), &value, sizeof(float));

    uint64_t total_writes = 0;
    for (int core = 0; core < num_cores_; ++core) {
        uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
        for (uint32_t pre = 0; pre < N; ++pre) {
            for (uint32_t post = 0; post < N; ++post) {
                uint64_t offset = static_cast<uint64_t>(pre) * N + post;
                uint64_t addr = base + offset * sizeof(float);
                auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, /*posted*/true);
                memory_->sendUntimedData(w);
                total_writes++;
            }
        }
        output_->verbose(CALL_INFO, 2, 0, "   核心%d: base=%" PRIu64 " 写入 %u x %u\n", core, base, N, N);
    }
    output_->verbose(CALL_INFO, 1, 0, "✅ WeightLoader发出写请求数=%" PRIu64 "\n", total_writes);
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

void WeightLoader::issueWritesForCoreFloats(int core, const std::vector<float>& wbuf) {
    if (!memory_) return;
    const uint32_t N = neurons_per_core_;
    const uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
    const size_t expected = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (validate_length_ && wbuf.size() < expected) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d权重长度不足(%zu<%zu)，用fill_value补齐\n", core, wbuf.size(), expected);
    }
    // 逐元素写（后续可做chunk聚合）
    size_t idx = 0;
    for (uint32_t pre = 0; pre < N; ++pre) {
        for (uint32_t post = 0; post < N; ++post) {
            float val;
            if (idx < wbuf.size()) val = row_major_ ? wbuf[idx] : wbuf[pre + static_cast<size_t>(post) * N];
            else val = fill_value_;
            std::vector<uint8_t> data(sizeof(float));
            std::memcpy(data.data(), &val, sizeof(float));
            uint64_t offset = static_cast<uint64_t>(pre) * N + post;
            uint64_t addr = base + offset * sizeof(float);
            auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, /*posted*/true);
            memory_->sendUntimedData(w);
            idx++;
        }
    }
}

bool WeightLoader::loadSingleFileAllCores(const std::string& path, const std::string& fmt) {
    std::vector<float> all;
    if (!readFileAllFloats(path, fmt, all)) return false;
    const uint32_t N = neurons_per_core_;
    const size_t per_core = static_cast<size_t>(N) * static_cast<size_t>(N);
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
    const uint32_t N = neurons_per_core_;
    const size_t per_core = static_cast<size_t>(N) * static_cast<size_t>(N);
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
    output_->verbose(CALL_INFO, 1, 0, "✅ 按核心分文件加载完成: 模板 %s\n", tmpl.c_str());
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
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 运行时未找到权重文件，跳过加载\n");
    }
    loaded_ = true;
}

bool WeightLoader::loadPerCoreFilesRuntime(const std::string& tmpl, const std::string& fmt) {
    const uint32_t N = neurons_per_core_;
    const size_t per_core = static_cast<size_t>(N) * static_cast<size_t>(N);
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
    output_->verbose(CALL_INFO, 1, 0, "✅ 运行时按核心分文件加载完成: 模板 %s\n", tmpl.c_str());
    return true;
}

void WeightLoader::issueWritesForCoreFloatsRuntime(int core, const std::vector<float>& wbuf) {
    if (!memory_) return;
    const uint32_t N = neurons_per_core_;
    const uint64_t base = base_addr_start_ + static_cast<uint64_t>(core) * per_core_stride_;
    const size_t expected = static_cast<size_t>(N) * static_cast<size_t>(N);
    
    if (validate_length_ && wbuf.size() < expected) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 运行时核心%d权重长度不足(%zu<%zu)，用fill_value补齐\n", core, wbuf.size(), expected);
    }
    
    // 使用时钟驱动的写入而不是sendUntimedData
    size_t idx = 0;
    for (uint32_t pre = 0; pre < N; ++pre) {
        for (uint32_t post = 0; post < N; ++post) {
            float val;
            if (idx < wbuf.size()) val = row_major_ ? wbuf[idx] : wbuf[pre + static_cast<size_t>(post) * N];
            else val = fill_value_;
            
            std::vector<uint8_t> data(sizeof(float));
            std::memcpy(data.data(), &val, sizeof(float));
            uint64_t offset = static_cast<uint64_t>(pre) * N + post;
            uint64_t addr = base + offset * sizeof(float);
            
            // 使用非posted写入确保立即完成
            auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, false); // posted=false
            memory_->send(w);
            pending_writes_++;  // 增加待处理写入计数
            
            /*
            output_->verbose(CALL_INFO, 2, 0, "💾 运行时写入: core=%d pre=%u post=%u addr=0x%lx val=%.6f pending=%u\n", 
                            core, pre, post, addr, val, pending_writes_);
            */
            idx++;
        }
    }
    // output_->verbose(CALL_INFO, 2, 0, "✅ 运行时核心%d权重写入完成: %u x %u\n", core, N, N);
}


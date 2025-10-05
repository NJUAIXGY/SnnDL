// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent.cc: SnnPE SubComponent版本实现文件
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;

// === 静态共享路由缓存定义 ===
std::mutex SnnPESubComponent::s_route_cache_mtx_;
std::unordered_map<std::string, std::weak_ptr<const SnnPESubComponent::RouteMap>> SnnPESubComponent::s_route_cache_;

SnnPESubComponent::SnnPESubComponent(ComponentId_t id, Params& params)
    : SnnCoreAPI(id, params), parent_(nullptr) {
    
    // 读取配置参数
    core_id_ = params.find<int>("core_id", 0);
    total_cores_ = params.find<int>("total_cores", 8);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    num_neurons_ = params.find<uint32_t>("num_neurons", 64);
    v_thresh_ = params.find<float>("v_thresh", 1.0f);
    v_reset_ = params.find<float>("v_reset", 0.0f);
    v_rest_ = params.find<float>("v_rest", 0.0f);
    tau_mem_ = params.find<float>("tau_mem", 20.0f);
    t_ref_ = params.find<uint32_t>("t_ref", 2);
    base_addr_ = params.find<uint64_t>("base_addr", 0);
    node_id_ = params.find<uint32_t>("node_id", 0);
    verbose_ = params.find<int>("verbose", 0);
    enable_weight_fetch_ = params.find<int>("enable_weight_fetch", 0) != 0;
    write_weights_on_init_ = params.find<int>("write_weights_on_init", 1) != 0;
    memory_warmup_cycles_ = params.find<uint64_t>("memory_warmup_cycles", 1000);
    init_default_weight_ = params.find<float>("init_default_weight", 0.5f);
    max_outstanding_requests_ = params.find<uint32_t>("max_outstanding_requests", 16);
    max_cache_entries_ = params.find<uint32_t>("max_cache_entries", 4096);
    use_event_weight_fallback_ = params.find<int>("use_event_weight_fallback", 0) != 0;
    event_weight_fallback_warned_ = false;
    merge_read_cacheline_ = params.find<int>("merge_read_cacheline", 1) != 0;
    merge_read_row_ = params.find<int>("merge_read_row", 0) != 0;
    line_size_bytes_ = params.find<uint32_t>("line_size_bytes", 64);
    // 全网读取扩展参数
    weights_cols_ = params.find<uint32_t>("weights_cols", 0);
    std::string index_mode_str = params.find<std::string>("index_mode", "pre_row_post_col");
    use_post_row_pre_col_ = (index_mode_str == "post_row_pre_col");
    if (weights_cols_ == 0) weights_cols_ = num_neurons_; // 默认沿用旧行宽
    enable_detailed_map_log_ = params.find<int>("enable_detailed_map_log", 0) != 0;
    // 权重验证参数
    verify_weights_ = params.find<int>("verify_weights", 0) != 0;
    weight_verify_samples_ = params.find<uint32_t>("weight_verify_samples", 16);
    expected_weight_value_ = params.find<float>("expected_weight_value", 0.0f);
    verify_epsilon_ = params.find<float>("verify_epsilon", 1e-4f);
    verify_log_each_sample_ = params.find<int>("verify_log_each_sample", 0) != 0;
    verify_against_file_ = params.find<int>("verify_against_file", 0) != 0;
    verify_file_template_ = params.find<std::string>("verify_file_template", "");
    // 路由模式参数
    std::string routing_mode = params.find<std::string>("routing_mode", "fixed");
    routing_weight_driven_ = (routing_mode == "weight_driven");
    weights_template_ = params.find<std::string>("weights_template", "");
    total_nodes_cfg_ = params.find<uint32_t>("total_nodes", 16);
    routing_epsilon_ = params.find<float>("routing_epsilon", 1e-8f);
    routing_topk_ = params.find<uint32_t>("routing_topk", 0);
    routing_topk_per_pe_ = params.find<uint32_t>("routing_topk_per_pe", 0);
    route_exclude_self_pe_ = params.find<int>("route_exclude_self_pe", 0) != 0;
    route_layers_mask_ = params.find<std::string>("route_layers_mask", "");
    route_filter_warn_ = params.find<int>("route_filter_warn", 1) != 0;
    // 映射框架集成
    mapping_mode_ = params.find<std::string>("mapping_mode", "off");
    mapping_edges_file_ = params.find<std::string>("mapping_edges_file", "");
    mapping_csv_has_header_ = params.find<int>("mapping_csv_has_header", 1) != 0;
    mapping_csv_separator_ = params.find<std::string>("mapping_csv_separator", ",");
    mapping_assume_block_ids_ = params.find<int>("mapping_assume_block_ids", 1) != 0;
    // 解析层间许可掩码
    allowed_layer_edges_.clear();
    allow_all_layers_ = true;
    if (!route_layers_mask_.empty()) {
        allow_all_layers_ = false;
        auto mask = route_layers_mask_;
        // 统一大小写，分隔符支持逗号或分号
        for (auto &ch : mask) ch = (char)std::toupper((unsigned char)ch);
        std::vector<std::string> toks;
        size_t start = 0;
        for (size_t i = 0; i <= mask.size(); ++i) {
            if (i == mask.size() || mask[i] == ',' || mask[i] == ';') {
                if (i > start) toks.emplace_back(mask.substr(start, i - start));
                start = i + 1;
            }
        }
        auto layerId = [](const std::string& s)->int{
            if (s == "I") return 0;      // Input 0-3
            if (s == "H1") return 1;     // Hidden1 4-7
            if (s == "H2") return 2;     // Hidden2 8-11
            if (s == "O") return 3;      // Output 12-15
            return -1;
        };
        for (auto &t : toks) {
            size_t p = t.find('>');
            if (p == std::string::npos) continue;
            std::string a = t.substr(0, p);
            std::string b = t.substr(p+1);
            int la = layerId(a); int lb = layerId(b);
            if (la >= 0 && lb >= 0) {
                uint32_t key = ((uint32_t)la << 8) | (uint32_t)lb;
                allowed_layer_edges_.insert(key);
            }
        }
    }
    
    // 获取权重文件路径
    weights_file_path_ = params.find<std::string>("weights_file", "");

    // === Supervised-learning (Phase 1) params ===
    learning_enabled_   = params.find<int>("learning_enabled", 0) != 0;
    learn_window_cycles_ = params.find<uint64_t>("learn_window_cycles", 1000);
    record_membrane_    = params.find<int>("record_membrane", 0) != 0;
    record_spike_times_ = params.find<int>("record_spike_times", 1) != 0;
    surrogate_type_     = params.find<std::string>("surrogate_type", "superspike");
    surrogate_beta_     = params.find<float>("surrogate_beta", 5.0f);
    error_file_template_ = params.find<std::string>("error_file", "");
    grad_accum_limit_    = (size_t) params.find<uint32_t>("grad_accum_limit", 0);
    apply_writeback_     = params.find<int>("apply_writeback", 0) != 0;
    apply_every_n_windows_ = params.find<uint32_t>("apply_every_n_windows", 1);
    learning_rate_       = params.find<float>("learning_rate", 0.001f);
    weight_decay_        = params.find<float>("weight_decay", 0.0f);

    // 参数日志改至 setup 以避免构造早期潜在问题
    
    // 初始化输出对象
    output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    
    // output_->verbose(CALL_INFO, 1, 0, "🔧 初始化SnnPE SubComponent (核心%d, %u个神经元)\n", 
    //                 core_id_, num_neurons_);
    
    // 输出权重验证参数以便调试
    output_->verbose(CALL_INFO, 1, 0, "🔍 权重验证配置: verify_weights=%d, samples=%u, expected=%.3f, log_each=%d\n",
                     verify_weights_ ? 1 : 0, weight_verify_samples_, expected_weight_value_, verify_log_each_sample_ ? 1 : 0);
    
    // 初始化神经元状态（复用SnnPE逻辑）
    neuron_states_.resize(num_neurons_);
    for (uint32_t i = 0; i < num_neurons_; i++) {
        neuron_states_[i] = NeuronState(v_rest_);
    }
    
    // 初始化内存访问
    memory_link_ = nullptr;
    memory_ = nullptr;
    next_request_id_ = 1;

    // Initialize learning window
    window_start_cycle_ = 0;
    current_window_index_ = 0;
    if (learning_enabled_) {
        error_buffer_.assign(num_neurons_, 0.0f);
    }


    
    // 初始化统计变量
    total_cycles_ = 0;
    active_cycles_ = 0;
    boot_read_sent_ = false;
    boot_write_sent_ = false;
    delayed_read_counter_ = 0;
    delayed_read_triggered_ = false;
    weights_initialized_ = false;
    memory_ready_ = false;
    stat_spikes_received_ = nullptr;
    stat_spikes_generated_ = nullptr;
    stat_neurons_fired_ = nullptr;
    stat_memory_requests_ = nullptr;
    stat_weight_cache_hits_ = nullptr;
    stat_weight_cache_misses_ = nullptr;
    stat_merged_reads_rows_ = nullptr;
    stat_merged_reads_cls_ = nullptr;
    stat_weights_verify_count_ = nullptr;
    stat_weights_mismatch_count_ = nullptr;
    stat_weights_verify_sum_ = nullptr;
    
    // 初始化内部计数器
    count_spikes_received_ = 0;
    count_spikes_generated_ = 0;
    count_neurons_fired_ = 0;
    count_memory_requests_ = 0;
    
    // 配置时钟
    std::string clock_freq = "1GHz";
    registerClock(clock_freq, new Clock::Handler2<SnnPESubComponent,&SnnPESubComponent::clockTick>(this));
    
    // 立即注册统计，避免在调用 getStatistics 前指针为空
    initializeStatistics();

    // output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心%d初始化完成\n", core_id_);
}

// 生成共享路由缓存键：尽量覆盖会影响路由构建结果的所有参数
std::string SnnPESubComponent::buildRouteCacheKey() const {
    try {
        std::ostringstream ss;
        // 路由模式与输入来源
        ss << "mode=" << (routing_weight_driven_ ? "wd" : "fixed");
        ss << ";mapping_mode=" << mapping_mode_;
        ss << ";weights_tpl=" << weights_template_;
        ss << ";edges_file=" << mapping_edges_file_;
        ss << ";csv_hdr=" << (mapping_csv_has_header_ ? 1 : 0);
        ss << ";csv_sep=" << mapping_csv_separator_;
        ss << ";assume_blk=" << (mapping_assume_block_ids_ ? 1 : 0);
        // 过滤/裁剪参数
        ss << ";eps=" << routing_epsilon_;
        ss << ";topk=" << routing_topk_;
        ss << ";topk_pe=" << routing_topk_per_pe_;
        ss << ";ex_self=" << (route_exclude_self_pe_ ? 1 : 0);
        ss << ";layers=" << route_layers_mask_;
        // 维度/规模参数
        ss << ";total_nodes=" << total_nodes_cfg_;
        ss << ";rows(perPE)=" << num_neurons_;
        ss << ";cols(global)=" << weights_cols_;
        return ss.str();
    } catch (...) {
        return std::string();
    }
}

SnnPESubComponent::~SnnPESubComponent() {
    // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁SnnPE SubComponent核心%d\n", core_id_);
    
    // 清理脉冲队列
    while (!incoming_spikes_.empty()) {
        delete incoming_spikes_.front();
        incoming_spikes_.pop();
    }
    
    delete output_;
}

void SnnPESubComponent::setParentInterface(SnnPEParentInterface* parent) {
    parent_ = parent;
    // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置父级接口\n", core_id_);
}

void SnnPESubComponent::init(unsigned int phase) {
    // output_->verbose(CALL_INFO, 1, 0, "🔄 核心%d init phase %u\n", core_id_, phase);
    
    if (phase == 0) {
        // 初始化统计收集
        initializeStatistics();
        
        // 配置内存端口（可选，但不覆盖已设置的链接）
        if (!memory_link_) {
            memory_link_ = configureLink("mem_link");
            if (memory_link_) output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d配置mem_link\n", core_id_);
        }
        
        // 加载StandardMem接口（Python可通过槽位提供）
        memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
            "memory", ComponentInfo::SHARE_NONE,
            registerTimeBase("1ns"),
            new SST::Interfaces::StandardMem::Handler2<SnnPESubComponent, &SnnPESubComponent::handleMemoryResponse>(this));
        if (memory_) {
            // output_->verbose(CALL_INFO, 1, 0, "✅ 核心%d加载StandardMem成功\n", core_id_);
        } else {
            // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未加载StandardMem，将使用默认权重\n", core_id_);
        }

        // 路由表构建（可选，支持共享缓存）
        if (routing_weight_driven_) {
            bool ok = false;
            // 生成共享缓存key，尽量覆盖影响路由的所有参数
            std::string cache_key = buildRouteCacheKey();
            if (!cache_key.empty()) {
                std::shared_ptr<const RouteMap> hit;
                {
                    std::lock_guard<std::mutex> g(s_route_cache_mtx_);
                    auto it = s_route_cache_.find(cache_key);
                    if (it != s_route_cache_.end()) hit = it->second.lock();
                }
                if (hit) {
                    routes_shared_ = hit;
                    ok = true;
                    // 统计共享表条目总数
                    uint64_t total_entries = 0; for (auto &kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
                    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
                    output_->verbose(CALL_INFO, 1, 0, "🔁 命中共享路由缓存: 核心%d, 源条目=%zu, 总目的=%" PRIu64 "\n",
                                     core_id_, routes_shared_->size(), total_entries);
                }
            }

            if (!ok) {
                // 未命中缓存，则构建一次
                if (mapping_mode_ == "edges_csv" && !mapping_edges_file_.empty()) {
                    ok = buildRoutesFromEdgesCSV();
                } else {
                    ok = buildWeightDrivenRoutes();
                }

                if (!ok) {
                    output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d权重驱动路由构建失败，将回退fixed路由\n", core_id_);
                    routing_weight_driven_ = false;
                } else {
                    // 将本地表封装为共享指针，并写入缓存
                    auto built = std::make_shared<RouteMap>(routes_by_source_);
                    routes_shared_ = built;
                    // 统计
                    uint64_t total_entries = 0; for (auto &kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
                    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
                    output_->verbose(CALL_INFO, 1, 0, "✅ 核心%d构建共享路由: 源条目=%zu, 总目的=%" PRIu64 "\n",
                                     core_id_, routes_shared_->size(), total_entries);
                    // 发布到进程级缓存
                    std::string cache_key = buildRouteCacheKey();
                    if (!cache_key.empty()) {
                        std::lock_guard<std::mutex> g(s_route_cache_mtx_);
                        s_route_cache_[cache_key] = built;
                    }
                    // 释放本地副本以减内存占用（后续使用共享表）
                    routes_by_source_.clear();
                }
            }
        }

        // 加载用于验证的权重文件（可选）
        if (verify_against_file_ && !verify_file_template_.empty()) {
            std::string path = verify_file_template_;
            size_t pos = path.find("{pe:02d}");
            if (pos != std::string::npos) {
                char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", node_id_);
                path.replace(pos, 8, buf);
            } else {
                pos = path.find("{pe}");
                if (pos != std::string::npos) path.replace(pos, 4, std::to_string(node_id_));
            }
            std::ifstream fin(path, std::ios::binary);
            if (fin.good()) {
                fin.seekg(0, std::ios::end);
                std::streamsize bytes = fin.tellg();
                fin.seekg(0, std::ios::beg);
                if (bytes > 0 && (bytes % sizeof(float) == 0)) {
                    size_t count = static_cast<size_t>(bytes / sizeof(float));
                    verify_file_buf_.resize(count);
                    fin.read(reinterpret_cast<char*>(verify_file_buf_.data()), bytes);
                    verify_file_loaded_ = true;
                    output_->verbose(CALL_INFO, 1, 0, "✅ 验证文件加载完成: %s (floats=%zu)\n", path.c_str(), verify_file_buf_.size());
                } else {
                    output_->verbose(CALL_INFO, 1, 0, "⚠️ 验证文件尺寸异常: %s\n", path.c_str());
                }
            } else {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开验证文件: %s\n", path.c_str());
            }
        }
    }

    // 将 init 相位转发给 StandardMem，以建立地址映射与握手
    if (memory_) {
        memory_->init(phase);
    }

    // Default weight initialization disabled, relying on WeightLoader
    if (phase == 4) {
        // 所有init阶段结束，允许后续时钟中发起访问
        memory_ready_ = true;
        // 重置验证状态
        verify_started_ = false;
        verify_requested_ = 0;
        verify_completed_ = 0;
        verify_sum_ = 0.0;
        verify_mismatch_count_ = 0;
    }
}

void SnnPESubComponent::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d setup 进入\n", core_id_);
    // output_->verbose(CALL_INFO, 1, 0,
    //     "🧩 参数: init_default_weight=%.3f, fallback=%d, merge_row=%d, merge_cl=%d, line=%uB, base_addr=%" PRIu64 ", N=%u\n",
    //     init_default_weight_, use_event_weight_fallback_, merge_read_row_, merge_read_cacheline_, line_size_bytes_, base_addr_, num_neurons_);
    
    // 验证组件状态
    if (!parent_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心%d没有父级接口\n", core_id_);
    }
    // 注意：此处不直接发起内存访问，避免在setup阶段 MemLink 尚未建立时触发 memHierarchy fatal
    if (!memory_) {
        // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置StandardMem，检查是否有直接权重文件\n", core_id_);
        
        // 权重将由WeightLoader组件通过内存接口加载
        if (!weights_file_path_.empty()) {
            // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d权重文件路径: %s (将由WeightLoader加载)\n", core_id_, weights_file_path_.c_str());
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置权重文件，将使用默认权重\n", core_id_);
        }
    }

    // output_->verbose(CALL_INFO, 1, 0, "✅ SnnPE SubComponent核心%d setup完成\n", core_id_);
}

void SnnPESubComponent::finish() {
    // output_->verbose(CALL_INFO, 1, 0, "🏁 SnnPE SubComponent核心%d完成仿真\n", core_id_);
    
    // 输出统计信息（使用内部计数器获得正确值）
    output_->verbose(CALL_INFO, 1, 0, "📊 核心%d统计: 接收脉冲=%" PRIu64 ", 生成脉冲=%" PRIu64 ", 神经元发放=%" PRIu64 "\n",
                    core_id_, 
                    count_spikes_received_,
                    count_spikes_generated_,
                    count_neurons_fired_);

    if (verify_weights_) {
        output_->verbose(CALL_INFO, 1, 0, "🔍 权重验证: 完成=%u, 不匹配=%" PRIu64 ", 平均值=%.6f (期望=%.6f)\n",
                         verify_completed_, verify_mismatch_count_,
                         (verify_completed_ ? (verify_sum_ / verify_completed_) : 0.0), expected_weight_value_);
    }
}

bool SnnPESubComponent::clockTick(Cycle_t current_cycle) {
    total_cycles_++;
    bool has_activity = false;
    // Learning window boundary check (Phase 2)
    if (learning_enabled_ && learn_window_cycles_ > 0) {
        uint64_t win_idx = (uint64_t)(total_cycles_ / learn_window_cycles_);
        if (total_cycles_ == 1 || win_idx != current_window_index_) {
            onWindowBoundary_(win_idx);
            current_window_index_ = win_idx;
        }
    }
    
    // 调试权重验证状态 (仅在前几个周期输出)
    /*
    if (verify_weights_ && total_cycles_ < 10) {
        output_->verbose(CALL_INFO, 2, 0, "🔍 核心%d状态检查: verify_weights=%d, memory_link=%s, memory_ready=%d, cycles=%lu, warmup=%lu\n",
                        core_id_, verify_weights_ ? 1 : 0, memory_link_ ? "yes" : "no", memory_ready_ ? 1 : 0, 
                        total_cycles_, memory_warmup_cycles_);
    }
    */
    
    // 处理输入脉冲队列
    while (!incoming_spikes_.empty()) {
        SpikeEvent* spike = incoming_spikes_.front();
        incoming_spikes_.pop();
        
        processLocalSpike(spike);
        has_activity = true;
        
        delete spike;
    }
    
    // 启动后按需读取权重（受暖机周期与开关控制）
    if (enable_weight_fetch_ && memory_ && memory_ready_ && total_cycles_ >= memory_warmup_cycles_) {
        // 示例：周期性读取一个权重并累加到某个神经元上（验证通路）
        // 实际模型应在突触更新处调用 requestWeight
        if (!delayed_read_triggered_) {
            uint32_t pre = 0;
            uint32_t post = 0;
            requestWeight(pre, post, [this, pre, post](float w){
                if (!neuron_states_.empty()) {
                    neuron_states_[post % num_neurons_].v_mem += 0.0f; // 仅拉通读路径，不直接修改
                }
            });
            delayed_read_triggered_ = true;
        }
    }

    // 权重正确性验证：在暖机完成后进行固定次数采样读取，对比 expected_weight_value_
    if (verify_weights_ && memory_ && memory_ready_ && total_cycles_ >= memory_warmup_cycles_) {
        if (!verify_started_) {
            verify_started_ = true;
            output_->verbose(CALL_INFO, 1, 0, "🎯 核心%d权重验证启动: 周期=%lu, 暖机阈值=%lu\n", 
                            core_id_, total_cycles_, memory_warmup_cycles_);
        }
        // 每个周期发起至多一个样本，避免拥塞
        if (verify_completed_ < weight_verify_samples_ && verify_requested_ - verify_completed_ < max_outstanding_requests_) {
            uint32_t sample_idx = verify_requested_;
            // 采样若干 (row, col)
            uint32_t row = (sample_idx * 13) % num_neurons_;                // 本地目标行
            uint32_t col = use_post_row_pre_col_ ? ((sample_idx * 7) % std::max<uint32_t>(1, weights_cols_))
                                                 : ((sample_idx * 7) % num_neurons_);
            // 新模式传参：(pre_global=col, post_local=row)；旧模式：(pre_local=row, post_local=col)
            uint32_t arg0 = use_post_row_pre_col_ ? col : row;
            uint32_t arg1 = use_post_row_pre_col_ ? row : col;
            requestWeight(arg0, arg1, [this, row, col](float w){
                verify_completed_++;
                verify_sum_ += static_cast<double>(w);
                bool mismatch = false;
                if (verify_against_file_ && verify_file_loaded_) {
                    // 使用文件中的期望值（row-major: row*weights_cols_ + col）
                    uint64_t idx = static_cast<uint64_t>(row) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(col);
                    float expected = 0.0f;
                    if (idx < verify_file_buf_.size()) expected = verify_file_buf_[idx];
                    mismatch = (std::fabs(w - expected) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(FILE): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected, std::fabs(w-expected), (mismatch?"MISMATCH":"OK"));
                    }
                } else {
                    // 回退到常数期望（兼容旧行为）
                    mismatch = (std::fabs(w - expected_weight_value_) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(CONST): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected_weight_value_, std::fabs(w-expected_weight_value_), (mismatch?"MISMATCH":"OK"));
                    }
                }
                if (mismatch) verify_mismatch_count_++;
                // 详细调试权重读取值（禁用默认回调日志；仅当逐样本日志开启时输出）
                if (verify_log_each_sample_) {
                    output_->verbose(CALL_INFO, 2, 0,
                        "🔎 权重验证回调: core=%d row=%u col=%u value=%.6f sum=%.6f count=%u\n",
                        core_id_, row, col, w, verify_sum_, verify_completed_);
                }
                if (stat_weights_verify_count_) stat_weights_verify_count_->addData(1);
                if (verify_mismatch_count_ && stat_weights_mismatch_count_) stat_weights_mismatch_count_->addData(1);
                if (stat_weights_verify_sum_) stat_weights_verify_sum_->addData(verify_sum_);
            });
            verify_requested_++;
        }
    }

    // 更新神经元状态（复用SnnPE逻辑）
    updateNeuronStates();
    
    // 检查并触发脉冲（复用SnnPE逻辑）
    for (uint32_t i = 0; i < num_neurons_; i++) {
        checkAndFireSpike(i);
    }
    
    if (has_activity) {
        active_cycles_++;
    }
    
    return false;  // 继续时钟
}

void SnnPESubComponent::deliverSpike(SpikeEvent* spike) {
    if (!spike) return;
    
    output_->verbose(CALL_INFO, 4, 0, "📨 核心%d接收脉冲: 源全局ID=%u, 目标全局ID=%u, 目标神经元=%u, 权重%.3f\n",
                    core_id_, spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getDestinationNeuron(), spike->getWeight());
    
    // 将脉冲加入队列，在时钟周期中处理
    incoming_spikes_.push(spike);
    
    // 更新两种统计：SST统计对象和内部计数器
    stat_spikes_received_->addData(1);
    count_spikes_received_++;
    
    // Debug output disabled to prevent excessive logging
    // printf("DEBUG: SnnPESubComponent核心%d接收脉冲，内部计数器更新: count_spikes_received_=%lu\n", 
    //        core_id_, count_spikes_received_);
}

void SnnPESubComponent::setMemoryLink(SST::Link* link) {
    memory_link_ = link;
    
    // ★ 关键修正：直接使用提供的Link进行内存操作 ★
    if (memory_link_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接成功\n", core_id_);
        memory_ready_ = true;  // 标记内存已准备就绪
    } else {
        output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接失败 (link=nullptr)\n", core_id_);
        memory_ready_ = false;
    }
}

bool SnnPESubComponent::hasWork() const {
    return !incoming_spikes_.empty() || 
           std::any_of(neuron_states_.begin(), neuron_states_.end(),
                      [](const NeuronState& state) { return state.v_mem > 0.1f; });
}

double SnnPESubComponent::getUtilization() const {
    if (total_cycles_ == 0) return 0.0;
    return static_cast<double>(active_cycles_) / static_cast<double>(total_cycles_);
}

void SnnPESubComponent::getStatistics(std::map<std::string, uint64_t>& stats) const {
    // 使用内部计数器而不是getCollectionCount()来获取正确的累计值
    stats["spikes_received"] = count_spikes_received_;
    stats["spikes_generated"] = count_spikes_generated_;
    stats["neurons_fired"] = count_neurons_fired_;
    stats["memory_requests"] = count_memory_requests_;
    stats["total_cycles"] = total_cycles_;
    stats["active_cycles"] = active_cycles_;
}

// ===== 核心计算方法（复用SnnPE实现）=====

void SnnPESubComponent::updateNeuronStates() {
    // 复用SnnPE的神经元状态更新逻辑
    for (uint32_t i = 0; i < num_neurons_; i++) {
        auto& neuron = neuron_states_[i];
        
        // 处理不应期
        if (neuron.refractory_timer > 0) {
            neuron.refractory_timer--;
            continue;
        }
        
        // 应用泄漏动态
        applyLeak(i);
    }
}

void SnnPESubComponent::applyLeak(uint32_t neuron_idx) {
    // 复用SnnPE的泄漏实现
    if (neuron_idx >= num_neurons_) return;
    
    auto& neuron = neuron_states_[neuron_idx];
    
    if (neuron.v_mem > v_rest_) {
        // 指数泄漏
        neuron.v_mem = v_rest_ + (neuron.v_mem - v_rest_) * exp(-1.0f / tau_mem_);
    }
}

void SnnPESubComponent::checkAndFireSpike(uint32_t neuron_idx) {
    // 复用SnnPE的脉冲触发逻辑
    if (neuron_idx >= num_neurons_) return;
    
    auto& neuron = neuron_states_[neuron_idx];
    
    // Capture current membrane before potential reset for logging
    float v_before = neuron.v_mem;
    bool will_fire = neuron_model_ ? neuron_model_->shouldFire(neuron_idx, neuron)
                                   : (neuron.v_mem >= v_thresh_ && neuron.refractory_timer == 0);
    if (will_fire) {
        // 神经元发放脉冲
        if (neuron_model_) {
            neuron_model_->onFired(neuron_idx, neuron);
        } else {
            neuron.v_mem = v_reset_;
            neuron.refractory_timer = t_ref_;
        }
        neuron.last_spike_time = total_cycles_;

        // Phase 1: record spike timeline (lightweight, default on when learning enabled)
        if (learning_enabled_ && record_spike_times_) {
            uint32_t g_id = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
            // On demand: also record membrane at firing if requested
            float v_fire = record_membrane_ ? v_before : 0.0f;
            spike_history_.emplace_back(g_id, static_cast<uint64_t>(total_cycles_), v_fire);
        }
        
        stat_neurons_fired_->addData(1);
        stat_spikes_generated_->addData(1);
        count_neurons_fired_++;
        count_spikes_generated_++;
        
        output_->verbose(CALL_INFO, 3, 0, "🔥 核心%d神经元%d发放脉冲! v_mem=%.3f -> %.3f\n",
                        core_id_, neuron_idx, v_thresh_, v_reset_);
        
        uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
        if (routing_weight_driven_) {
            // 优先使用共享路由表，回退到本地表
            const RouteMap* route_tbl = routes_shared_ ? routes_shared_.get() : &routes_by_source_;
            auto itrt = route_tbl->find(source_global);
            if (itrt != route_tbl->end()) {
                const auto& dests = itrt->second;
                if (stat_fanout_per_spike_) stat_fanout_per_spike_->addData((uint64_t)dests.size());
                for (uint32_t dest_global : dests) {
                    uint32_t dest_node = dest_global / num_neurons_;
                    float output_weight = 1.0f;
                    SpikeEvent* output_spike = new SpikeEvent(
                        source_global,
                        dest_global,
                        dest_node,
                        output_weight,
                        total_cycles_);
                    if (parent_) parent_->sendSpike(output_spike); else delete output_spike;
                }
                output_->verbose(CALL_INFO, 2, 0, "🌐 权重驱动扇出: 源g=%u, 目的数=%zu\n", source_global, dests.size());
            } else {
                // 无路由目标，静默
            }
        } else {
            // 原固定层间映射
            uint32_t target_neuron = 0;
            uint32_t target_node = node_id_;
            float output_weight = 1.0f;
            if (node_id_ >= 0 && node_id_ <= 3) {
                uint32_t target_hidden_base = (node_id_ < 2) ? 4 : 8;
                uint32_t target_hidden_node = target_hidden_base + (node_id_ % 2) * 2 + (neuron_idx % 2);
                target_node = target_hidden_node;
                target_neuron = target_hidden_node * 16 + neuron_idx;
                output_->verbose(CALL_INFO, 2, 0, "🔥 输入层节点%d神经元%d -> 隐藏层节点%d神经元%d\n",
                                 node_id_, neuron_idx, target_node, target_neuron);
            } else if (node_id_ >= 4 && node_id_ <= 11) {
                uint32_t target_output_node = 12 + ((node_id_ - 4) / 2);
                target_node = target_output_node;
                target_neuron = target_output_node * 16 + (neuron_idx % 16);
                output_->verbose(CALL_INFO, 2, 0, "🔥 隐藏层节点%d神经元%d -> 输出层节点%d神经元%d\n",
                                 node_id_, neuron_idx, target_node, target_neuron);
            } else {
                output_->verbose(CALL_INFO, 2, 0, "🔥 输出层节点%d神经元%d发放，不发送外部脉冲\n",
                                 node_id_, neuron_idx);
                return;
            }
            SpikeEvent* output_spike = new SpikeEvent(
                source_global,
                target_neuron,
                target_node,
                output_weight,
                total_cycles_);
            if (parent_) parent_->sendSpike(output_spike); else delete output_spike;
        }
    }
}

// === Phase 2: Learning window handling and error loading ===
void SnnPESubComponent::onWindowBoundary_(uint64_t window_idx) {
    // For now, just (re)load error buffer from file each window if template provided.
    // Future: could switch to per-epoch/batch semantics.
    (void)window_idx;
    loadErrorsForWindow_(window_idx);
    // Optional: prune gradient map if exceeds cap
    if (grad_accum_limit_ > 0 && local_grad_.size() > grad_accum_limit_) {
        // Minimal policy: clear all to keep memory bounded
        local_grad_.clear();
    }
    // Apply updates periodically if enabled
    if (apply_writeback_ && apply_every_n_windows_ > 0 && (window_idx % apply_every_n_windows_ == 0)) {
        if (memory_ && memory_ready_) {
            applyLocalWeightUpdates_();
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 学习: 写回启用但内存接口不可用，跳过本窗写回\n");
        }
    }
}

void SnnPESubComponent::loadErrorsForWindow_(uint64_t window_idx) {
    if (!learning_enabled_) return;
    if (error_file_template_.empty()) {
        // No file specified -> keep zeros (do nothing)
        if (error_buffer_.size() != num_neurons_) error_buffer_.assign(num_neurons_, 0.0f);
        return;
    }
    std::string path = replacePlaceholders_(error_file_template_);
    // Also support {win} placeholder
    {
        size_t p = path.find("{win}");
        if (p != std::string::npos) {
            std::string w = std::to_string(window_idx);
            path.replace(p, 5, w);
        }
    }
    std::ifstream fin(path);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 学习: 无法打开误差文件 %s, 本窗使用0误差\n", path.c_str());
        if (error_buffer_.size() != num_neurons_) error_buffer_.assign(num_neurons_, 0.0f);
        else std::fill(error_buffer_.begin(), error_buffer_.end(), 0.0f);
        return;
    }
    // Parse: supports lines of "id val" or single float per line (indexed by line)
    std::vector<float> tmp(num_neurons_, 0.0f);
    std::string line;
    uint32_t line_idx = 0;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        uint32_t id = UINT32_MAX; float val = 0.0f;
        if ((iss >> id >> val)) {
            if (id < num_neurons_) tmp[id] = val;
        } else {
            iss.clear(); iss.str(line);
            if (iss >> val) {
                if (line_idx < num_neurons_) tmp[line_idx] = val;
                line_idx++;
            }
        }
    }
    error_buffer_.swap(tmp);
}

void SnnPESubComponent::applyLocalWeightUpdates_() {
    if (!apply_writeback_) return;
    if (local_grad_.empty()) return;
    const uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    const size_t bytes_per_float = sizeof(float);
    uint64_t total_writes = 0;

    // Iterate over accumulated gradients and issue per-weight writes.
    // Note: this minimal implementation only updates weights present in cache
    // to compute absolute new values; others are skipped to avoid corrupting memory.
    size_t skipped_uncached = 0;
    for (const auto& kv : local_grad_) {
        uint64_t key = kv.first;
        float grad = kv.second;
        uint32_t row = (uint32_t)(key / width);
        uint32_t col = (uint32_t)(key % width);
        (void)row; (void)col;

        auto itc = weight_cache_.find(key);
        if (itc == weight_cache_.end()) {
            skipped_uncached++;
            continue; // skip if not in cache
        }
        float old_w = itc->second;
        float new_w = old_w - learning_rate_ * grad;
        if (weight_decay_ != 0.0f) {
            new_w -= weight_decay_ * old_w; // simple L2 decay
        }

        uint64_t offset = key; // linear index within matrix
        uint64_t addr = base_addr_ + offset * bytes_per_float;
        std::vector<uint8_t> data(bytes_per_float);
        std::memcpy(data.data(), &new_w, bytes_per_float);
        auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, false);
        memory_->send(w);
        if (stat_memory_requests_) stat_memory_requests_->addData(1);
        total_writes++;

        // Update cache with new value
        itc->second = new_w;
    }
    // Clear gradients after applying
    local_grad_.clear();
    output_->verbose(CALL_INFO, 1, 0, "📝 学习: 写回完成 writes=%" PRIu64 ", 跳过(未缓存)=%zu\n", total_writes, skipped_uncached);
}

void SnnPESubComponent::processLocalSpike(SpikeEvent* spike_event) {
    // 复用SnnPE的本地脉冲处理逻辑
    if (!spike_event) return;
    
    uint32_t dest = spike_event->getDestinationNeuron();
    uint32_t target_neuron = dest;
    // 全局ID → 本地ID 映射
    if (dest >= num_neurons_) {
        if (dest >= global_neuron_base_ && dest < global_neuron_base_ + num_neurons_) {
            target_neuron = static_cast<uint32_t>(dest - global_neuron_base_);
            output_->verbose(CALL_INFO, 4, 0, "🔁 核心%d将全局ID%d映射为本地ID%d\n", core_id_, dest, target_neuron);
        } else {
            output_->verbose(CALL_INFO, 2, 0, "⚠️ 核心%d收到无法映射的目标神经元%d的脉冲\n", core_id_, dest);
            return;
        }
    }
    
    auto& neuron = neuron_states_[target_neuron];
    
    // 检查是否在不应期
    if (neuron.refractory_timer > 0) {
        output_->verbose(CALL_INFO, 4, 0, "⚠️ 核心%d神经元%d在不应期，忽略脉冲\n", 
                        core_id_, target_neuron);
        return;
    }
    
    // 使用权重缓存/按需读取
    float weight = 0.0f;
    bool have_mem_weight = false;
    if (enable_weight_fetch_ && memory_ && memory_ready_) {
        // 计算本地/全局索引
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t pre_local = 0;
        uint32_t post_local = 0;
        // 目标始终映射为本地行
        if (post_global >= global_neuron_base_ && post_global < global_neuron_base_ + num_neurons_) {
            post_local = static_cast<uint32_t>(post_global - global_neuron_base_);
        } else {
            post_local = target_neuron; // 已在上方完成映射
        }
        // 源映射（仅旧模式需要本地pre_local）；新模式直接使用pre_global作为列
        if (pre_global >= global_neuron_base_ && pre_global < global_neuron_base_ + num_neurons_) {
            pre_local = static_cast<uint32_t>(pre_global - global_neuron_base_);
        } else {
            uint64_t pe_base = static_cast<uint64_t>(global_neuron_base_) - static_cast<uint64_t>(core_id_) * static_cast<uint64_t>(num_neurons_);
            pre_local = static_cast<uint32_t>((static_cast<uint64_t>(pre_global) - pe_base) % static_cast<uint64_t>(num_neurons_));
        }

        // 选择键值与请求参数
        uint64_t key = 0;
        uint32_t req_pre_param = 0;   // requestWeight 第一个参数
        uint32_t req_post_param = 0;  // requestWeight 第二个参数

        if (use_post_row_pre_col_) {
            // 新模式：行=post_local，列=pre_global
            key = static_cast<uint64_t>(post_local) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(pre_global);
            req_pre_param = pre_global;   // 列
            req_post_param = post_local;  // 行
        } else {
            // 旧模式：行=pre_local，列=post_local
            key = static_cast<uint64_t>(pre_local) * static_cast<uint64_t>(num_neurons_) + post_local;
            req_pre_param = pre_local;    // 行
            req_post_param = post_local;  // 列
        }

        auto it = weight_cache_.find(key);
        if (it != weight_cache_.end()) {
            weight = it->second;
            have_mem_weight = true;
            if (stat_weight_cache_hits_) stat_weight_cache_hits_->addData(1);
            if (!first_cache_hit_logged_) {
                if (use_post_row_pre_col_) {
                    output_->verbose(CALL_INFO, 2, 0, "🟢 首次命中(全网): row(post_l)=%u, col(pre_g)=%u, key=%" PRIu64 ", weight=%.3f\n",
                                     post_local, pre_global, key, weight);
                } else {
                    output_->verbose(CALL_INFO, 2, 0, "🟢 首次命中: pre_l=%u, post_l=%u, key=%" PRIu64 ", weight=%.3f\n",
                                     pre_local, post_local, key, weight);
                }
                first_cache_hit_logged_ = true;
            }
        } else if (outstanding_requests_ < max_outstanding_requests_) {
            outstanding_requests_++;
            if (outstanding_requests_ > pending_reqs_peak_) pending_reqs_peak_ = outstanding_requests_;
            requestWeight(req_pre_param, req_post_param, [this, key](float w){
                // 简单容量限制
                if (weight_cache_.size() >= max_cache_entries_) {
                    weight_cache_.clear();
                }
                weight_cache_[key] = w;
                if (outstanding_requests_ > 0) outstanding_requests_--;
            });
            if (stat_weight_cache_misses_) stat_weight_cache_misses_->addData(1);
            if (!first_cache_miss_logged_) {
                if (use_post_row_pre_col_) {
                    output_->verbose(CALL_INFO, 2, 0, "🟡 首次未命中并发起读(全网): row(post_l)=%u, col(pre_g)=%u, key=%" PRIu64 "\n",
                                     post_local, pre_global, key);
                } else {
                    output_->verbose(CALL_INFO, 2, 0, "🟡 首次未命中并发起读: pre_l=%u, post_l=%u, key=%" PRIu64 "\n",
                                     pre_local, post_local, key);
                }
                first_cache_miss_logged_ = true;
            }
        }
    }
    if (!have_mem_weight) {
        // 回退策略：可选择使用事件权重，或直接使用默认初始权重（与内存一致）
        if (use_event_weight_fallback_) {
            weight = spike_event->getWeight();
            if (!event_weight_fallback_warned_) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d启用事件权重回退，事件权重=%.3f\n", core_id_, weight);
                event_weight_fallback_warned_ = true;
            }
        } else {
            weight = 0.0f;
        }
    }
    if (neuron_model_) {
        neuron_model_->onSynapticEvent(target_neuron, weight, neuron);
    } else {
        neuron.v_mem += weight;
    }

    // Phase 2: online gradient accumulation using windowed error
    if (learning_enabled_) {
        const float err = (target_neuron < error_buffer_.size()) ? error_buffer_[target_neuron] : 0.0f;
        if (err != 0.0f) {
            const float sgrad = computeSurrogateGrad_(neuron.v_mem);
            const float contrib = err * sgrad; // pre spike counts as 1 event
            uint64_t key = 0;
            if (use_post_row_pre_col_) {
                const uint32_t width = weights_cols_;
                key = (uint64_t)target_neuron * (uint64_t)width + (uint64_t)spike_event->getSourceNeuron();
                local_grad_[key] += contrib;
            } else {
                // Only when pre belongs to this core (pre_local resolvable)
                const uint32_t pre_global = spike_event->getSourceNeuron();
                if (pre_global >= global_neuron_base_ && pre_global < global_neuron_base_ + num_neurons_) {
                    const uint32_t pre_local = (uint32_t)(pre_global - global_neuron_base_);
                    const uint32_t width = num_neurons_;
                    key = (uint64_t)pre_local * (uint64_t)width + (uint64_t)target_neuron;
                    local_grad_[key] += contrib;
                }
            }
            if (grad_accum_limit_ > 0 && local_grad_.size() > grad_accum_limit_) {
                local_grad_.clear();
            }
        }
    }

    // 一次性详细日志：打印全局/本地映射与地址
    if (enable_detailed_map_log_ || !detailed_log_emitted_) {
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t pre_local_dbg = (pre_global>=global_neuron_base_ && pre_global<global_neuron_base_+num_neurons_)
                                 ? static_cast<uint32_t>(pre_global - global_neuron_base_)
                                 : 0u;
        uint32_t post_local_dbg = target_neuron;
        uint64_t offset_dbg = 0;
        if (use_post_row_pre_col_) {
            offset_dbg = static_cast<uint64_t>(post_local_dbg) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(pre_global);
        } else {
            offset_dbg = static_cast<uint64_t>(pre_local_dbg) * static_cast<uint64_t>(num_neurons_) + post_local_dbg;
        }
        uint64_t addr_dbg = base_addr_ + offset_dbg * sizeof(float);
        output_->verbose(CALL_INFO, 1, 0,
            "🧪 详细权重调试: 事件权重=%.3f, 内存权重=%s, 最终权重=%.3f, 回退=%s\n",
            spike_event->getWeight(), have_mem_weight ? "有" : "无", weight, use_event_weight_fallback_ ? "启用" : "禁用");
        output_->verbose(CALL_INFO, 1, 0,
            "🧪 一次性详细映射: pre_g=%u->pre_l=%u, post_g=%u->post_l=%u, base=%" PRIu64 ", off=%" PRIu64 ", addr=%" PRIu64 ", weight=%.3f\n",
            pre_global, pre_local_dbg, post_global, post_local_dbg, base_addr_, offset_dbg, addr_dbg, weight);
        detailed_log_emitted_ = true;
    }
    output_->verbose(CALL_INFO, 5, 0, "⚡ 核心%d神经元%d: v_mem=%.3f (添加权重%.3f)\n",
                    core_id_, target_neuron, neuron.v_mem, weight);
    
    // 检查是否达到阈值并发放脉冲
    checkAndFireSpike(target_neuron);
}

void SnnPESubComponent::requestWeight(uint32_t pre_neuron, uint32_t post_neuron, 
                                    std::function<void(float)> callback) {
    // 地址映射：
    //  - 旧模式(pre_row_post_col):  row=pre_local, col=post_local,   width=num_neurons_
    //  - 新模式(post_row_pre_col): row=post_local, col=pre_global,  width=weights_cols_
    uint32_t row = 0, col = 0;
    uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    if (use_post_row_pre_col_) {
        col = pre_neuron;  // 传入的 pre_neuron 在新模式下表示源全局ID
        row = post_neuron; // 传入的 post_neuron 表示目标本地ID
    } else {
        row = pre_neuron;
        col = post_neuron;
    }
    uint64_t offset = static_cast<uint64_t>(row) * static_cast<uint64_t>(width) + static_cast<uint64_t>(col);
    uint64_t addr = base_addr_ + offset * sizeof(float);

    if (!memory_) {
        // 无StandardMem，直接返回默认权重
        if (callback) callback(0.5f);
        return;
    }

    // 生成读取请求
    // 合并策略
    // 生成读取请求
    uint32_t target_row = row;     // 行起点
    uint32_t target_col = col;     // 列
    uint32_t bytes_per_float = sizeof(float);
    uint64_t request_addr = addr;
    size_t request_size = sizeof(float);
    bool is_row = false;
    uint32_t col_start = target_col;
    uint32_t count_floats = 1;

    if (merge_read_row_) {
        is_row = true;
        col_start = 0;
        count_floats = width;
        request_addr = base_addr_ + static_cast<uint64_t>(target_row) * width * bytes_per_float;
        request_size = static_cast<size_t>(count_floats) * bytes_per_float;
        if (stat_merged_reads_rows_) stat_merged_reads_rows_->addData(1);
    } else if (merge_read_cacheline_) {
        uint32_t floats_per_line = std::max<uint32_t>(1, line_size_bytes_ / bytes_per_float);
        col_start = (target_col / floats_per_line) * floats_per_line;
        count_floats = std::min<uint32_t>(floats_per_line, width - col_start);
        request_addr = base_addr_ + (static_cast<uint64_t>(target_row) * width + col_start) * bytes_per_float;
        request_size = static_cast<size_t>(count_floats) * bytes_per_float;
        if (stat_merged_reads_cls_) stat_merged_reads_cls_->addData(1);
    }

    output_->verbose(CALL_INFO, 4, 0, "📤 读请求: row=%u, col=%u, is_row=%d, col_start=%u, count=%u, addr=%" PRIu64 ", size=%zu\n",
                     target_row, target_col, is_row, col_start, count_floats, request_addr, request_size);
    auto* read = new SST::Interfaces::StandardMem::Read(request_addr, request_size);
    uint64_t reqId = read->getID();
    PendingMemoryRequest pmr;
    pmr.request_id = reqId;
    pmr.address = request_addr;
    pmr.size = request_size;
    pmr.is_row = is_row;
    pmr.pre = target_row;          // 存放行索引
    pmr.post_start = col_start;    // 存放列起点
    pmr.count_floats = count_floats;
    pmr.has_single_cb = (callback != nullptr);
    pmr.cb_post = target_col;      // 单元素回调的列索引
    pmr.single_cb = callback;
    pending_memory_requests_[reqId] = pmr;
    memory_->send(read);
    stat_memory_requests_->addData(1);
    count_memory_requests_++;
}

void SnnPESubComponent::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
    if (!req) return;
    
    output_->verbose(CALL_INFO, 4, 0, "📨 核心%d收到内存响应: ID=%" PRIu64 "\n", 
                    core_id_, req->getID());
    
    // 查找对应的挂起请求
    auto it = pending_memory_requests_.find(req->getID());
    if (it != pending_memory_requests_.end()) {
        PendingMemoryRequest pending_req = it->second; // 拷贝一份，便于先erase
        pending_memory_requests_.erase(it);

        auto* readResp = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
        if (readResp && !readResp->data.empty()) {
            const std::vector<uint8_t>& bytes = readResp->data;
            // 将返回数据拆成float并填入缓存
            size_t float_count = bytes.size() / sizeof(float);
            const float* fptr = reinterpret_cast<const float*>(bytes.data());
            
            // 详细调试读取的字节数据
            output_->verbose(CALL_INFO, 3, 0, "📥 内存响应: addr=0x%lx, bytes=%zu, floats=%zu\n",
                              pending_req.address, bytes.size(), float_count);
            if (float_count > 0 && float_count <= 4) {
                output_->verbose(CALL_INFO, 3, 0, "   原始字节: ");
                for (size_t b = 0; b < std::min(bytes.size(), (size_t)16); b++) {
                    printf("%02x ", bytes[b]);
                }
                printf("\n");
                output_->verbose(CALL_INFO, 3, 0, "   解析浮点: ");
                for (size_t f = 0; f < float_count; f++) {
                    printf("%.6f ", fptr[f]);
                }
                printf("\n");
            }
            
            for (size_t i = 0; i < float_count; ++i) {
                uint32_t col_idx = pending_req.post_start + static_cast<uint32_t>(i);
                uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
                if (col_idx >= width) break;
                uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + col_idx;
                // 容量控制
                if (weight_cache_.size() >= max_cache_entries_) {
                    weight_cache_.clear();
                }
                weight_cache_[key] = fptr[i];
                output_->verbose(CALL_INFO, 4, 0, "   缓存权重: row=%u col=%u key=%lu value=%.6f\n",
                                  pending_req.pre, col_idx, key, fptr[i]);
            }
            output_->verbose(CALL_INFO, 4, 0, "📥 合并读填充: row=%u, col_start=%u, count=%zu\n",
                              pending_req.pre, pending_req.post_start, float_count);
            // 单目标回调（如果需要）
            if (pending_req.has_single_cb && pending_req.single_cb) {
                uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
                uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + pending_req.cb_post;
                float value = 0.0f;
                auto itc = weight_cache_.find(key);
                if (itc != weight_cache_.end()) value = itc->second;
                pending_req.single_cb(value);
            }
        } else {
            // 无数据时，触发回退回调
            if (pending_req.has_single_cb && pending_req.single_cb) {
                pending_req.single_cb(0.0f);
            }
        }
        // 合并读：统一在响应时递减并发计数
        if (outstanding_requests_ > 0) outstanding_requests_--;
    }
    
    delete req;
}

void SnnPESubComponent::initializeStatistics() {
    // output_->verbose(CALL_INFO, 2, 0, "📊 核心%d初始化统计收集\n", core_id_);
    
    stat_spikes_received_ = registerStatistic<uint64_t>("spikes_received");
    stat_spikes_generated_ = registerStatistic<uint64_t>("spikes_generated");
    stat_neurons_fired_ = registerStatistic<uint64_t>("neurons_fired");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_weight_cache_hits_ = registerStatistic<uint64_t>("weight_cache_hits");
    stat_weight_cache_misses_ = registerStatistic<uint64_t>("weight_cache_misses");
    stat_merged_reads_rows_ = registerStatistic<uint64_t>("merged_reads_rows");
    stat_merged_reads_cls_ = registerStatistic<uint64_t>("merged_reads_cls");
    stat_weights_verify_count_ = registerStatistic<uint64_t>("weights_verify_count");
    stat_weights_mismatch_count_ = registerStatistic<uint64_t>("weights_mismatch_count");
    stat_weights_verify_sum_ = registerStatistic<double>("weights_verify_sum");
    // 扩展统计
    stat_routes_entries_ = registerStatistic<uint64_t>("routes_entries");
    stat_fanout_per_spike_ = registerStatistic<uint64_t>("fanout_per_spike");
    
    // output_->verbose(CALL_INFO, 2, 0, "✅ 核心%d统计收集初始化完成\n", core_id_);
}
bool SnnPESubComponent::buildWeightDrivenRoutes() {
    // 需要 weights_template_ 包含 {pe} 占位符；需要 weights_cols_ 和 num_neurons_ 定义行/列
    if (weights_template_.empty()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建失败：weights_template 未提供\n");
        return false;
    }
    routes_by_source_.clear();
    const uint32_t rows = num_neurons_;          // 每PE行数（本地目标神经元数）
    const uint32_t cols = weights_cols_;         // 全网列数（全局源神经元数）
    const uint32_t total_nodes = total_nodes_cfg_;
    const size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    // 临时结构：pre_global -> list of (abs(weight), dest_global)
    std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> tmp;
    tmp.reserve(cols);
    uint64_t dropped_self_pe = 0;
    uint64_t dropped_layer_mask = 0;
    // 遍历所有PE的权重文件，建立 pre_global -> 目的候选列表
    for (uint32_t pe = 0; pe < total_nodes; ++pe) {
        // 生成路径
        std::string path = weights_template_;
        // 支持 {pe:02d} 和 {pe}
        size_t pos = path.find("{pe:02d}");
        if (pos != std::string::npos) {
            char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", pe);
            path.replace(pos, 8, buf);
        } else {
            pos = path.find("{pe}");
            if (pos != std::string::npos) path.replace(pos, 4, std::to_string(pe));
        }
        std::ifstream fin(path, std::ios::binary);
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：无法读取权重文件 %s\n", path.c_str());
            continue;
        }
        fin.seekg(0, std::ios::end);
        std::streamsize bytes = fin.tellg();
        fin.seekg(0, std::ios::beg);
        if (bytes <= 0 || (bytes % sizeof(float) != 0)) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件尺寸异常 %s\n", path.c_str());
            continue;
        }
        size_t count = static_cast<size_t>(bytes / sizeof(float));
        if (count < expected) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件过短 %s (%zu<%zu)\n", path.c_str(), count, expected);
            continue;
        }
        std::vector<float> buf(count);
        fin.read(reinterpret_cast<char*>(buf.data()), bytes);
        // 行优先：row-major，index = row*cols + col
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                size_t idx = static_cast<size_t>(row) * static_cast<size_t>(cols) + col;
                float w = buf[idx];
                if (std::fabs(w) > routing_epsilon_) {
                    uint32_t pre_global = col;
                    uint32_t dest_global = pe * rows + row; // 每PE的全局基按 rows 间隔
                    // 过滤：同PE
                    if (route_exclude_self_pe_) {
                        uint32_t src_pe = pre_global / rows;
                        if (src_pe == pe) { dropped_self_pe++; continue; }
                    }
                    // 过滤：层间掩码
                    if (!allow_all_layers_) {
                        uint32_t src_pe = pre_global / rows;
                        uint32_t la = getLayerIdFromPE(src_pe);
                        uint32_t lb = getLayerIdFromPE(pe);
                        uint32_t key = (la<<8) | lb;
                        if (allowed_layer_edges_.find(key) == allowed_layer_edges_.end()) { dropped_layer_mask++; continue; }
                    }
                    tmp[pre_global].emplace_back(std::fabs(w), dest_global);
                }
            }
        }
    }
    // 对每个源应用Top-K裁剪（全局或按PE）
    for (auto &kv : tmp) {
        uint32_t pre = kv.first;
        auto &lst = kv.second;
        if (lst.empty()) continue;

        if (routing_topk_per_pe_ > 0) {
            // 按目的PE分组，每组保留Top-K
            std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> by_pe;
            by_pe.reserve(16);
            for (auto &p : lst) {
                uint32_t dest_global = p.second;
                uint32_t pe = dest_global / rows;
                by_pe[pe].push_back(p);
            }
            std::vector<uint32_t> out;
            for (auto &g : by_pe) {
                auto &vec = g.second;
                if (vec.size() > routing_topk_per_pe_) {
                    std::partial_sort(vec.begin(), vec.begin()+routing_topk_per_pe_, vec.end(),
                        [](const auto& a, const auto& b){ return a.first > b.first; });
                    vec.resize(routing_topk_per_pe_);
                }
                for (auto &p : vec) out.push_back(p.second);
            }
            // 如果还需要全局Top-K
            if (routing_topk_ > 0 && out.size() > routing_topk_) {
                // 需要回看权重，故再次从lst中过滤
                std::vector<std::pair<float,uint32_t>> tmp2;
                tmp2.reserve(out.size());
                // 建立集合以快速查重
                std::unordered_set<uint32_t> keep(out.begin(), out.end());
                for (auto &p : lst) if (keep.count(p.second)) tmp2.push_back(p);
                std::partial_sort(tmp2.begin(), tmp2.begin()+routing_topk_, tmp2.end(),
                    [](const auto& a, const auto& b){ return a.first > b.first; });
                tmp2.resize(routing_topk_);
                std::vector<uint32_t> final_out; final_out.reserve(tmp2.size());
                for (auto &p : tmp2) final_out.push_back(p.second);
                routes_by_source_[pre] = std::move(final_out);
            } else {
                routes_by_source_[pre] = std::move(out);
            }
        } else if (routing_topk_ > 0) {
            if (lst.size() > routing_topk_) {
                std::partial_sort(lst.begin(), lst.begin()+routing_topk_, lst.end(),
                    [](const auto& a, const auto& b){ return a.first > b.first; });
                lst.resize(routing_topk_);
            }
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            routes_by_source_[pre] = std::move(out);
        } else {
            // 无裁剪：仅去重
            std::sort(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
            lst.erase(std::unique(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second==b.second; }), lst.end());
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            routes_by_source_[pre] = std::move(out);
        }
    }
    // 统计与提醒
    uint64_t total_entries = 0;
    for (auto &kv : routes_by_source_) total_entries += (uint64_t)kv.second.size();
    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
    if (route_exclude_self_pe_ || !allow_all_layers_) {
        if (route_filter_warn_) {
            output_->verbose(CALL_INFO, 1, 0,
                "⚠️ 路由过滤已启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                route_exclude_self_pe_ ? 1 : 0, route_layers_mask_.c_str(), dropped_self_pe, dropped_layer_mask);
        } else {
            output_->verbose(CALL_INFO, 2, 0,
                "路由过滤: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                route_exclude_self_pe_ ? 1 : 0, route_layers_mask_.c_str(), dropped_self_pe, dropped_layer_mask);
        }
    }
    return !routes_by_source_.empty();
}

uint32_t SnnPESubComponent::getLayerIdFromPE(uint32_t pe) const {
    // 固定4x4网格层划分：I:0-3, H1:4-7, H2:8-11, O:12-15
    if (pe <= 3) return 0;
    if (pe <= 7) return 1;
    if (pe <= 11) return 2;
    return 3;
}

bool SnnPESubComponent::buildRoutesFromEdgesCSV() {
    routes_by_source_.clear();
    const uint32_t rows = num_neurons_;
    std::ifstream fin(mapping_edges_file_);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开映射边文件: %s\n", mapping_edges_file_.c_str());
        return false;
    }
    std::string line;
    if (mapping_csv_has_header_) std::getline(fin, line);
    auto split = [this](const std::string& s)->std::vector<std::string>{
        std::vector<std::string> out; std::string cur; char sep = mapping_csv_separator_.empty() ? ',' : mapping_csv_separator_[0];
        std::istringstream ss(s);
        while (std::getline(ss, cur, sep)) out.push_back(cur);
        if (out.empty()) { std::istringstream ss2(s); while (ss2 >> cur) out.push_back(cur); }
        return out;
    };
    std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> tmp;
    uint64_t dropped_self = 0, dropped_layer = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto toks = split(line);
        if (toks.size() < 2) continue;
        uint32_t src = (uint32_t) std::stoul(toks[0]);
        uint32_t dst = (uint32_t) std::stoul(toks[1]);
        float w = 1.0f;
        if (toks.size() >= 3) { try { w = std::stof(toks[2]); } catch(...) { w = 1.0f; } }
        if (std::fabs(w) <= routing_epsilon_) continue;
        if (mapping_assume_block_ids_) {
            uint32_t src_pe = src / rows;
            uint32_t dst_pe = dst / rows;
            if (route_exclude_self_pe_ && src_pe == dst_pe) { dropped_self++; continue; }
            if (!allow_all_layers_) {
                uint32_t la = getLayerIdFromPE(src_pe);
                uint32_t lb = getLayerIdFromPE(dst_pe);
                uint32_t key = (la<<8) | lb;
                if (allowed_layer_edges_.find(key) == allowed_layer_edges_.end()) { dropped_layer++; continue; }
            }
        }
        tmp[src].emplace_back(std::fabs(w), dst);
    }
    for (auto &kv : tmp) {
        uint32_t pre = kv.first;
        auto &lst = kv.second;
        if (lst.empty()) continue;
        if (routing_topk_per_pe_ > 0) {
            std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> by_pe;
            by_pe.reserve(16);
            for (auto &p : lst) {
                uint32_t pe = mapping_assume_block_ids_ ? (p.second / rows) : 0;
                by_pe[pe].push_back(p);
            }
            std::vector<uint32_t> out;
            for (auto &g : by_pe) {
                auto &vec = g.second;
                if (vec.size() > routing_topk_per_pe_) {
                    std::partial_sort(vec.begin(), vec.begin()+routing_topk_per_pe_, vec.end(),
                        [](const auto& a, const auto& b){ return a.first > b.first; });
                    vec.resize(routing_topk_per_pe_);
                }
                for (auto &p : vec) out.push_back(p.second);
            }
            if (routing_topk_ > 0 && out.size() > routing_topk_) {
                std::vector<std::pair<float,uint32_t>> tmp2; tmp2.reserve(out.size());
                std::unordered_set<uint32_t> keep(out.begin(), out.end());
                for (auto &p : lst) if (keep.count(p.second)) tmp2.push_back(p);
                std::partial_sort(tmp2.begin(), tmp2.begin()+routing_topk_, tmp2.end(),
                    [](const auto& a, const auto& b){ return a.first > b.first; });
                tmp2.resize(routing_topk_);
                std::vector<uint32_t> final_out; final_out.reserve(tmp2.size());
                for (auto &p : tmp2) final_out.push_back(p.second);
                routes_by_source_[pre] = std::move(final_out);
            } else {
                routes_by_source_[pre] = std::move(out);
            }
        } else if (routing_topk_ > 0) {
            if (lst.size() > routing_topk_) {
                std::partial_sort(lst.begin(), lst.begin()+routing_topk_, lst.end(),
                    [](const auto& a, const auto& b){ return a.first > b.first; });
                lst.resize(routing_topk_);
            }
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            routes_by_source_[pre] = std::move(out);
        } else {
            std::sort(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
            lst.erase(std::unique(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second==b.second; }), lst.end());
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            routes_by_source_[pre] = std::move(out);
        }
    }
    uint64_t total_entries = 0; for (auto &kv : routes_by_source_) total_entries += (uint64_t)kv.second.size();
    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
    if ((route_exclude_self_pe_ || !allow_all_layers_) && route_filter_warn_) {
        output_->verbose(CALL_INFO, 1, 0,
            "⚠️ 路由过滤(映射CSV)启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self=%" PRIu64 ", layer=%" PRIu64 ")\n",
            route_exclude_self_pe_?1:0, route_layers_mask_.c_str(), dropped_self, dropped_layer);
    }
    return !routes_by_source_.empty();
}

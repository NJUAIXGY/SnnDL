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

using namespace SST;
using namespace SST::SnnDL;

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
    enable_detailed_map_log_ = params.find<int>("enable_detailed_map_log", 0) != 0;
    // 权重验证参数
    verify_weights_ = params.find<int>("verify_weights", 0) != 0;
    weight_verify_samples_ = params.find<uint32_t>("weight_verify_samples", 16);
    expected_weight_value_ = params.find<float>("expected_weight_value", 0.0f);
    verify_epsilon_ = params.find<float>("verify_epsilon", 1e-4f);
    verify_log_each_sample_ = params.find<int>("verify_log_each_sample", 0) != 0;
    
    // 获取权重文件路径
    weights_file_path_ = params.find<std::string>("weights_file", "");

    // 参数日志改至 setup 以避免构造早期潜在问题
    
    // 初始化输出对象
    output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    
    output_->verbose(CALL_INFO, 1, 0, "🔧 初始化SnnPE SubComponent (核心%d, %u个神经元)\n", 
                    core_id_, num_neurons_);
    
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

    output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心%d初始化完成\n", core_id_);
}

SnnPESubComponent::~SnnPESubComponent() {
    output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁SnnPE SubComponent核心%d\n", core_id_);
    
    // 清理脉冲队列
    while (!incoming_spikes_.empty()) {
        delete incoming_spikes_.front();
        incoming_spikes_.pop();
    }
    
    delete output_;
}

void SnnPESubComponent::setParentInterface(SnnPEParentInterface* parent) {
    parent_ = parent;
    output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置父级接口\n", core_id_);
}

void SnnPESubComponent::init(unsigned int phase) {
    output_->verbose(CALL_INFO, 1, 0, "🔄 核心%d init phase %u\n", core_id_, phase);
    
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
            output_->verbose(CALL_INFO, 1, 0, "✅ 核心%d加载StandardMem成功\n", core_id_);
        } else {
            // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未加载StandardMem，将使用默认权重\n", core_id_);
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
    output_->verbose(CALL_INFO, 1, 0, "🏁 SnnPE SubComponent核心%d完成仿真\n", core_id_);
    
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
    
    // 调试权重验证状态 (仅在前几个周期输出)
    if (verify_weights_ && total_cycles_ < 10) {
        output_->verbose(CALL_INFO, 2, 0, "🔍 核心%d状态检查: verify_weights=%d, memory_link=%s, memory_ready=%d, cycles=%lu, warmup=%lu\n",
                        core_id_, verify_weights_ ? 1 : 0, memory_link_ ? "yes" : "no", memory_ready_ ? 1 : 0, 
                        total_cycles_, memory_warmup_cycles_);
    }
    
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
            // 均匀选择若干 (pre, post)
            uint32_t pre = (sample_idx * 7) % num_neurons_;
            uint32_t post = (sample_idx * 13) % num_neurons_;
            requestWeight(pre, post, [this, pre, post](float w){
                verify_completed_++;
                verify_sum_ += static_cast<double>(w);
                if (std::fabs(w - expected_weight_value_) > verify_epsilon_) {
                    verify_mismatch_count_++;
                }
                // 详细调试权重读取值
                output_->verbose(CALL_INFO, 1, 0,
                    "🔎 权重验证回调: core=%d pre=%u post=%u value=%.6f sum=%.6f count=%u\n",
                    core_id_, pre, post, w, verify_sum_, verify_completed_);
                if (verify_log_each_sample_) {
                    output_->verbose(CALL_INFO, 1, 0,
                        "🔎 权重样本: core=%d pre=%u post=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                        core_id_, pre, post, w, expected_weight_value_, std::fabs(w-expected_weight_value_),
                        (std::fabs(w-expected_weight_value_)<=verify_epsilon_?"OK":"MISMATCH"));
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
        output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接成功\n", core_id_);
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
    
    if (neuron.v_mem >= v_thresh_ && neuron.refractory_timer == 0) {
        // 神经元发放脉冲
        neuron.v_mem = v_reset_;
        neuron.refractory_timer = t_ref_;
        neuron.last_spike_time = total_cycles_;
        
        stat_neurons_fired_->addData(1);
        stat_spikes_generated_->addData(1);
        count_neurons_fired_++;
        count_spikes_generated_++;
        
        output_->verbose(CALL_INFO, 3, 0, "🔥 核心%d神经元%d发放脉冲! v_mem=%.3f -> %.3f\n",
                        core_id_, neuron_idx, v_thresh_, v_reset_);
        
        // 创建输出脉冲 - 基于频率分类网络连接模式
        uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
        
        // 确定目标神经元和节点基于网络层次结构
        uint32_t target_neuron = 0;
        uint32_t target_node = node_id_;
        float output_weight = 0.0f;
        
        // 网络连接模式：
        // 输入层 (节点0-3, 神经元0-7) -> 隐藏层 (节点4-11, 神经元8-39)
        // 隐藏层 (节点4-11, 神经元8-39) -> 输出层 (节点12-15, 神经元40-47)
        
        if (node_id_ >= 0 && node_id_ <= 3) {
            // 输入层 -> 隐藏层
            // 扇出连接：每个输入节点连接到多个隐藏层节点，激活所有8个隐藏层节点
            // 连接模式: 节点0,1->4,5,6,7; 节点2,3->8,9,10,11
            uint32_t target_hidden_base = (node_id_ < 2) ? 4 : 8;  // 前两个输入节点连到4-7，后两个连到8-11
            uint32_t target_hidden_node = target_hidden_base + (node_id_ % 2) * 2 + (neuron_idx % 2);  
            target_node = target_hidden_node;
            target_neuron = 8 + (target_hidden_node - 4) * 4 + neuron_idx;  // 隐藏层神经元8-39
            
            output_->verbose(CALL_INFO, 2, 0, "🔥 输入层节点%d神经元%d -> 隐藏层节点%d神经元%d\n",
                           node_id_, neuron_idx, target_node, target_neuron);
        } else if (node_id_ >= 4 && node_id_ <= 11) {
            // 隐藏层 -> 输出层  
            // 简化连接：每两个隐藏层节点连接到一个输出层节点
            uint32_t target_output_node = 12 + ((node_id_ - 4) / 2);  // 节点4,5->12; 6,7->13; 8,9->14; 10,11->15
            target_node = target_output_node;
            target_neuron = 40 + (target_output_node - 12) * 2 + (neuron_idx % 2);  // 输出层神经元40-47
            
            output_->verbose(CALL_INFO, 2, 0, "🔥 隐藏层节点%d神经元%d -> 输出层节点%d神经元%d\n",
                           node_id_, neuron_idx, target_node, target_neuron);
        } else {
            // 输出层不发送外部脉冲
            output_->verbose(CALL_INFO, 2, 0, "🔥 输出层节点%d神经元%d发放，不发送外部脉冲\n",
                           node_id_, neuron_idx);
            return;
        }

        SpikeEvent* output_spike = new SpikeEvent(
            source_global,   // 源神经元（全局ID）
            target_neuron,   // 目标神经元（全局ID）
            target_node,     // 目标节点（本PE内）
            output_weight,   // 权重
            total_cycles_    // 时间戳
        );
        
        // 通过父级接口发送脉冲
        if (parent_) {
            parent_->sendSpike(output_spike);
        } else {
            delete output_spike;
        }
    }
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
        // 计算 pre_local 与 post_local
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t pre_local = 0;
        uint32_t post_local = 0;
        // 全局→本地：使用本核的 global_neuron_base_ 做基准
        if (pre_global >= global_neuron_base_ && pre_global < global_neuron_base_ + num_neurons_) {
            pre_local = static_cast<uint32_t>(pre_global - global_neuron_base_);
        } else {
            // 若源不在本核，本核的权重矩阵仍以本核 pre 为行索引，
            // 此处若需要跨核权重，应使用源所在核的 base_addr 发起读取。
            // 当前多核PE设计为每核自有权重块，因此以源核读取为准。
            // 为了通用性，先按环内常见映射：取源所在核在本PE内的相对索引区间折算。
            uint64_t pe_base = static_cast<uint64_t>(global_neuron_base_) - static_cast<uint64_t>(core_id_) * static_cast<uint64_t>(num_neurons_);
            pre_local = static_cast<uint32_t>((static_cast<uint64_t>(pre_global) - pe_base) % static_cast<uint64_t>(num_neurons_));
        }
        if (post_global >= global_neuron_base_ && post_global < global_neuron_base_ + num_neurons_) {
            post_local = static_cast<uint32_t>(post_global - global_neuron_base_);
        } else {
            post_local = target_neuron; // 已在上方完成映射
        }
        uint64_t key = static_cast<uint64_t>(pre_local) * static_cast<uint64_t>(num_neurons_) + post_local;
        auto it = weight_cache_.find(key);
        if (it != weight_cache_.end()) {
            weight = it->second;
            have_mem_weight = true;
            if (stat_weight_cache_hits_) stat_weight_cache_hits_->addData(1);
            if (!first_cache_hit_logged_) {
                output_->verbose(CALL_INFO, 2, 0, "🟢 首次命中: pre_l=%u, post_l=%u, key=%" PRIu64 ", weight=%.3f\n",
                                 pre_local, post_local, key, weight);
                first_cache_hit_logged_ = true;
            }
        } else if (outstanding_requests_ < max_outstanding_requests_) {
            outstanding_requests_++;
            if (outstanding_requests_ > pending_reqs_peak_) pending_reqs_peak_ = outstanding_requests_;
            requestWeight(pre_local, post_local, [this, key](float w){
                // 简单容量限制
                if (weight_cache_.size() >= max_cache_entries_) {
                    weight_cache_.clear();
                }
                weight_cache_[key] = w;
                if (outstanding_requests_ > 0) outstanding_requests_--;
            });
            if (stat_weight_cache_misses_) stat_weight_cache_misses_->addData(1);
            if (!first_cache_miss_logged_) {
                output_->verbose(CALL_INFO, 2, 0, "🟡 首次未命中并发起读: pre_l=%u, post_l=%u, key=%" PRIu64 "\n",
                                 pre_local, post_local, key);
                first_cache_miss_logged_ = true;
            }
        }
    }
    if (!have_mem_weight) {
        // 回退策略：可选择使用事件权重，或直接使用默认初始权重（与内存一致）
        if (use_event_weight_fallback_) {
            weight = spike_event->getWeight();
            if (!event_weight_fallback_warned_) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d启用事件权重回退，优先级低于内存权重且仅在未命中时使用\n", core_id_);
                event_weight_fallback_warned_ = true;
            }
        } else {
            weight = 0.0f;
        }
    }
    neuron.v_mem += weight;
    
    // 一次性详细日志：打印全局/本地映射与地址
    if (enable_detailed_map_log_ || !detailed_log_emitted_) {
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t pre_local_dbg = (pre_global>=global_neuron_base_ && pre_global<global_neuron_base_+num_neurons_)
                                 ? static_cast<uint32_t>(pre_global - global_neuron_base_)
                                 : 0u;
        uint32_t post_local_dbg = target_neuron;
        uint64_t offset_dbg = static_cast<uint64_t>(pre_local_dbg) * static_cast<uint64_t>(num_neurons_) + post_local_dbg;
        uint64_t addr_dbg = base_addr_ + offset_dbg * sizeof(float);
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
    // 简化地址映射：base_addr + (pre*num_neurons + post)*sizeof(float)
    uint64_t offset = static_cast<uint64_t>(pre_neuron) * static_cast<uint64_t>(num_neurons_) + post_neuron;
    uint64_t addr = base_addr_ + offset * sizeof(float);

    if (!memory_) {
        // 无StandardMem，直接返回默认权重
        if (callback) callback(0.5f);
        return;
    }

    // 生成读取请求
    // 合并策略
    uint32_t target_pre = pre_neuron;
    uint32_t target_post = post_neuron;
    uint32_t bytes_per_float = sizeof(float);
    uint64_t request_addr = addr;
    size_t request_size = sizeof(float);
    bool is_row = false;
    uint32_t post_start = target_post;
    uint32_t count_floats = 1;

    if (merge_read_row_) {
        is_row = true;
        post_start = 0;
        count_floats = num_neurons_;
        request_addr = base_addr_ + static_cast<uint64_t>(target_pre) * num_neurons_ * bytes_per_float;
        request_size = static_cast<size_t>(count_floats) * bytes_per_float;
        if (stat_merged_reads_rows_) stat_merged_reads_rows_->addData(1);
    } else if (merge_read_cacheline_) {
        uint32_t floats_per_line = std::max<uint32_t>(1, line_size_bytes_ / bytes_per_float);
        post_start = (target_post / floats_per_line) * floats_per_line;
        count_floats = std::min<uint32_t>(floats_per_line, num_neurons_ - post_start);
        request_addr = base_addr_ + (static_cast<uint64_t>(target_pre) * num_neurons_ + post_start) * bytes_per_float;
        request_size = static_cast<size_t>(count_floats) * bytes_per_float;
        if (stat_merged_reads_cls_) stat_merged_reads_cls_->addData(1);
    }

    output_->verbose(CALL_INFO, 4, 0, "📤 读请求: pre=%u, post=%u, is_row=%d, post_start=%u, count=%u, addr=%" PRIu64 ", size=%zu\n",
                     target_pre, target_post, is_row, post_start, count_floats, request_addr, request_size);
    auto* read = new SST::Interfaces::StandardMem::Read(request_addr, request_size);
    uint64_t reqId = read->getID();
    PendingMemoryRequest pmr;
    pmr.request_id = reqId;
    pmr.address = request_addr;
    pmr.size = request_size;
    pmr.is_row = is_row;
    pmr.pre = target_pre;
    pmr.post_start = post_start;
    pmr.count_floats = count_floats;
    pmr.has_single_cb = (callback != nullptr);
    pmr.cb_post = target_post;
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
                uint32_t post_idx = pending_req.post_start + static_cast<uint32_t>(i);
                if (post_idx >= num_neurons_) break;
                uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(num_neurons_) + post_idx;
                // 容量控制
                if (weight_cache_.size() >= max_cache_entries_) {
                    weight_cache_.clear();
                }
                weight_cache_[key] = fptr[i];
                output_->verbose(CALL_INFO, 4, 0, "   缓存权重: pre=%u post=%u key=%lu value=%.6f\n",
                                  pending_req.pre, post_idx, key, fptr[i]);
            }
            output_->verbose(CALL_INFO, 4, 0, "📥 合并读填充: pre=%u, post_start=%u, count=%zu\n",
                              pending_req.pre, pending_req.post_start, float_count);
            // 单目标回调（如果需要）
            if (pending_req.has_single_cb && pending_req.single_cb) {
                uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(num_neurons_) + pending_req.cb_post;
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
    output_->verbose(CALL_INFO, 2, 0, "📊 核心%d初始化统计收集\n", core_id_);
    
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
    
    output_->verbose(CALL_INFO, 2, 0, "✅ 核心%d统计收集初始化完成\n", core_id_);
}
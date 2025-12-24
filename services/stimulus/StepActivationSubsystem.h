// -*- c++ -*-
//
// StepActivationSubsystem:
// - Step 级随机激活（选择 pre → 生成 fanout spikes → 注入到本 PE / 外部 PE）
// - 可选：基于 BCSR reachability 的路由采样（step_activation_use_bcsr_routes=1）
//
// 目标（Phase3）：
// - 将 Step 注入的“事务逻辑/BCSR 解析/调度状态”从 MultiCorePE 下沉为独立子系统；
// - MultiCorePE 仅作为控制壳：负责把时钟/阶段事件转发给子系统，并提供最小注入回调。

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace SST { class Output; }
namespace SST { namespace Statistics { template <typename T> class Statistic; } }

namespace SST { namespace SnnDL {

class SpikeEvent;
class INocTransport;

class StepActivationSubsystem final {
public:
    struct Config {
        bool enable = false;
        double fraction = 0.0;
        uint32_t fanout = 0;
        uint64_t seed = 0xdecafbadULL;

        // 0=BeginGather 触发；>0=固定周期（cycle）
        uint64_t period_cycles = 0;
        // BeginGather 触发的 leader core（默认 0；-1 表示任意核心）
        int trigger_core = 0;

        bool reset_mem_each_step = false;
        double event_weight = 0.0;  // 预留（当前未参与注入权重；保持兼容）

        // BCSR reachability 路由采样（仅影响“post 选择”）
        bool use_bcsr_routes = false;
        std::string bcsr_template;
        uint32_t bcsr_rows_per_core = 0;
        uint32_t bcsr_br = 16;
        uint32_t bcsr_bc = 16;
        uint32_t bcsr_idx_bytes = 2;
        uint32_t bcsr_val_bytes = 4;
        uint64_t bcsr_rowptr_offset = 0;
        uint64_t bcsr_colidx_offset = 0;
        uint64_t bcsr_blockdata_offset = 0;
        uint64_t bcsr_blockids_offset = 0;
        double bcsr_weight_epsilon = 0.0;
        bool log_enable = false;
        bool build_local_only = true;
        uint64_t bcsr_align = 64;
    };

    struct Runtime {
        SST::Output* log = nullptr;
        int node_id = 0;
        int total_nodes = 1;
        uint64_t global_neuron_base = 0;
        int num_cores = 1;
        int neurons_per_core = 1;
        uint32_t neurons_per_pe_cfg = 0;
        bool sentinel_enabled = false;
        long step_diag_cap_cfg = 0;
        int step_diag_enable_cfg = 0;

        // NoC 抽象接口（Phase4-A1.3）：优先使用该接口进行注入/外发
        INocTransport* noc = nullptr;

        // 注入回调（由 MultiCorePE 提供，保持“直达目标 core / 外部 NIC”语义不变）
        std::function<void(int /*core_id*/, SpikeEvent*)> deliver_to_core;
        std::function<void(SpikeEvent*)> send_external;
        std::function<void()> reset_membranes;
    };

    struct Stats {
        SST::Statistics::Statistic<uint64_t>* invocations = nullptr;
        SST::Statistics::Statistic<uint64_t>* pre_selected = nullptr;
        SST::Statistics::Statistic<uint64_t>* spike_attempts = nullptr;
        SST::Statistics::Statistic<uint64_t>* spikes_injected = nullptr;
        SST::Statistics::Statistic<uint64_t>* route_hits = nullptr;
        SST::Statistics::Statistic<uint64_t>* route_misses = nullptr;
        SST::Statistics::Statistic<uint64_t>* local_drops = nullptr;
    };

    void configure(const Config& cfg);
    void bindRuntime(const Runtime& rt);
    void bindStats(const Stats& st);

    // 仅当 enable && use_bcsr_routes 时加载；失败会自动降级 use_bcsr_routes=false。
    void initBcsrReachabilityIfEnabled();

    // NIC 完成 init 后置为 true；若存在 pending 注入，将在后续 tick 中执行。
    void setInjectionReady(bool ready) { injection_ready_ = ready; }

    // 由 MultiCorePE 每拍调用：处理 pending 注入与固定周期注入。
    void tick(uint64_t current_cycle);

    // 由 MultiCorePE 的阶段事件转发：BeginGather（仅当 period_cycles==0 时有效）
    void onBeginGather(uint32_t seq, uint64_t ts_ns, int core_id);

    // 由 MultiCorePE 的阶段事件转发：EndScatter（用于 step_reset_mem_each_step）
    void onEndScatter(uint32_t seq);

private:
    int determineTargetUnit_(uint32_t neuron_id) const;
    void injectStepActivations_(uint32_t seq, uint64_t sim_time_ns);

    // BCSR reachability build helpers
    std::string formatBcsrPath_(int pe, int core) const;
    uint64_t alignUp_(uint64_t value, uint64_t align) const;
    bool computeBcsrOffsets_(uint32_t n_block_rows, uint32_t total_blocks,
                             uint64_t block_bytes,
                             uint64_t& rowptr_offset, uint64_t& colidx_offset,
                             uint64_t& blockdata_offset, uint64_t& blockids_offset) const;
    bool checkBcsrOffsets_(uint64_t file_size, uint32_t n_block_rows,
                           uint32_t total_blocks, uint64_t block_bytes,
                           uint64_t& rowptr_offset, uint64_t& colidx_offset,
                           uint64_t& blockdata_offset, uint64_t& blockids_offset) const;
    bool buildRoutesFromBcsrFile_(const std::string& path, uint32_t pe_id, uint32_t core_index);
    bool loadBcsrReachability_();
    void computeRouteRatios_() const;

    Config cfg_{};
    Runtime rt_{};
    Stats st_{};

    bool injection_ready_ = false;
    bool pending_step_inject_ = false;
    uint32_t pending_step_seq_ = 0;
    uint64_t pending_step_ts_ns_ = 0;

    uint64_t next_cycle_ = 0;
    uint32_t fixed_seq_ = 1;
    uint32_t last_injection_seq_ = std::numeric_limits<uint32_t>::max();
    uint32_t last_reset_seq_ = std::numeric_limits<uint32_t>::max();
    uint32_t seq_warn_count_ = 0;

    bool route_diag_done_ = false;
    bool route_ack_logged_ = false;
    bool route_warned_ = false;

    std::vector<std::vector<uint32_t>> step_routes_;
    std::vector<uint32_t> pre_with_routes_;
};

}} // namespace SST::SnnDL

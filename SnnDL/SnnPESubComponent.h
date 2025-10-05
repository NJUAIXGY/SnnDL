#ifndef _H_SST_SNN_PE_SUBCOMPONENT
#define _H_SST_SNN_PE_SUBCOMPONENT

#include <sst/core/subcomponent.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <queue>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>
#include <string>
#include <vector>
#include "SpikeEvent.h"
#include "SnnPEParentInterface.h"
#include "SnnCoreAPI.h"
#include "SnnNeuronModel.h"

namespace SST {
namespace SnnDL {

class SnnPESubComponent : public SnnCoreAPI {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        SnnPESubComponent,
        "SnnDL",
        "SnnPESubComponent",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "SNN Processing Element SubComponent",
        SST::SnnDL::SnnPESubComponent
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "ID of the core", ""},
        {"total_cores", "Total number of cores in the PE", "8"},
        {"global_neuron_base", "Global base ID for neurons in this core", "0"},
        {"num_neurons", "Number of neurons in this core", "64"},
        {"v_thresh", "Neuron threshold voltage", "1.0"},
        {"v_reset", "Neuron reset voltage", "0.0"},
        {"v_rest", "Neuron resting voltage", "0.0"},
        {"tau_mem", "Membrane time constant", "20.0"},
        {"t_ref", "Refractory period in clock cycles", "2"},
        {"base_addr", "Base address for weight fetching", "0"},
        {"node_id", "Node ID of the parent PE", "0"},
        {"verbose", "Verbosity level", "0"},
        {"enable_weight_fetch", "Enable fetching weights from memory", "0"},
        {"write_weights_on_init", "Write default weights to memory on init", "1"},
        {"memory_warmup_cycles", "Cycles to wait before starting memory operations", "1000"},
        {"init_default_weight", "Default weight value to initialize memory with", "0.5"},
        {"max_outstanding_requests", "Maximum number of outstanding memory requests", "16"},
        {"max_cache_entries", "Maximum number of entries in the weight cache", "4096"},
        {"use_event_weight_fallback", "Use weight from spike event if memory fetch fails", "0"},
        {"merge_read_cacheline", "Merge memory reads to cache line size", "1"},
        {"merge_read_row", "Merge memory reads to a full row", "0"},
        {"line_size_bytes", "Cache line size in bytes", "64"},
        {"weights_cols", "Number of columns in weight matrix when using global read (post_row_pre_col)", "0"},
        {"index_mode", "Indexing mode: pre_row_post_col (default) or post_row_pre_col", "pre_row_post_col"},
        {"enable_detailed_map_log", "Enable detailed logging of neuron mapping", "0"},
        {"verify_weights", "Enable weight verification", "0"},
        {"weight_verify_samples", "Number of weight samples to verify", "16"},
        {"expected_weight_value", "Expected weight value for verification", "0.0"},
        {"verify_epsilon", "Epsilon for floating point comparison", "1e-4"},
        {"verify_log_each_sample", "Log each weight sample for verification", "0"},
        {"verify_against_file", "Verify weights by comparing memory reads against weight file (0/1)", "0"},
        {"verify_file_template", "Template for per-PE weight files used for verification (e.g., .../classification_weights_pe_{pe}.bin)", ""},
        // 路由模式（默认fixed：沿用内置层间映射；weight_driven：按权重文件驱动扇出）
        {"routing_mode", "Routing mode: fixed (default) or weight_driven", "fixed"},
        {"weights_template", "Template for per-PE weight files, e.g. .../classification_weights_pe_{pe}.bin", ""},
        {"total_nodes", "Total number of PEs/nodes in the system", "16"},
        {"routing_epsilon", "Threshold to treat a weight as non-zero when building routes", "1e-8"},
        {"routing_topk", "Global top-K destinations per source (0=unlimited)", "0"},
        {"routing_topk_per_pe", "Top-K destinations per destination-PE per source (0=unlimited)", "0"},
        // 路由过滤：层间/同PE
        {"route_exclude_self_pe", "Exclude routes targeting the same PE as source (0/1)", "0"},
        {"route_layers_mask", "Allowed layer transitions, e.g. I>H1,H1>H2,H2>O", ""},
        {"route_filter_warn", "Print prominent warning when route filters are enabled (0/1)", "1"}
        ,
        // 映射框架集成（默认关闭）
        {"mapping_mode", "Mapping-driven routes: off (default) or edges_csv", "off"},
        {"mapping_edges_file", "CSV file of edges: src_global,dst_global,weight(optional)", ""},
        {"mapping_csv_has_header", "Edges CSV has header line (0/1)", "1"},
        {"mapping_csv_separator", "Edges CSV separator (default ',')", ","},
        {"mapping_assume_block_ids", "Assume global_id=pe*rows+row for PE inference (0/1)", "1"}
        ,
        // === Supervised Learning (minimal, default off) ===
        {"learning_enabled", "Enable supervised learning features (0/1)", "0"},
        {"learn_window_cycles", "Learning window size in clock cycles", "1000"},
        {"record_membrane", "Record membrane potentials (lightweight; 0/1)", "0"},
        {"record_spike_times", "Record spike times for learning (0/1)", "1"},
        {"surrogate_type", "Surrogate gradient type: superspike|sigmoid|piecewise", "superspike"},
        {"surrogate_beta", "Surrogate gradient beta (steepness)", "5.0"},
        {"error_file", "Per-core error file template (supports {node},{core})", ""},
        {"grad_accum_limit", "Cap on gradient entries before pruning (0=unlimited)", "0"},
        // === Writeback (Phase 3, default off) ===
        {"apply_writeback", "Apply gradient updates to DRAM weights (0/1)", "0"},
        {"apply_every_n_windows", "Apply updates every N learning windows", "1"},
        {"learning_rate", "Learning rate for SGD updates", "0.001"},
        {"weight_decay", "L2 weight decay coefficient (requires cached weight)", "0.0"}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"spikes_received", "Number of spikes received by this core", "spikes", 1},
        {"spikes_generated", "Number of spikes generated by this core", "spikes", 1},
        {"neurons_fired", "Number of times neurons in this core fired", "events", 1},
        {"memory_requests", "Number of memory requests sent", "requests", 1},
        {"weight_cache_hits", "Number of weight cache hits", "hits", 1},
        {"weight_cache_misses", "Number of weight cache misses", "misses", 1},
        {"merged_reads_rows", "Number of memory reads merged to a full row", "requests", 1},
        {"merged_reads_cls", "Number of memory reads merged to a cache line", "requests", 1},
        {"weights_verify_count", "Number of weights verified", "count", 1},
        {"weights_mismatch_count", "Number of weights that failed verification", "count", 1},
        {"weights_verify_sum", "Sum of verified weights for averaging", "value", 1},
        {"routes_entries", "Total number of route entries built for weight-driven routing", "count", 1},
        {"fanout_per_spike", "Fanout size per emitted spike in weight-driven routing", "count", 1}
    )

    SnnPESubComponent(SST::ComponentId_t id, SST::Params& params);
    ~SnnPESubComponent();

    virtual void setParentInterface(SnnPEParentInterface* parent);
    virtual void init(unsigned int phase) override;
    virtual void setup() override;
    virtual void finish() override;

    virtual void deliverSpike(SpikeEvent* spike) override;
    virtual bool hasWork() const override;
    virtual double getUtilization() const override;
    virtual void getStatistics(std::map<std::string, uint64_t>& stats) const override;
    void setMemoryLink(SST::Link* link);

private:
    // === Minimal supervised-learning state (Phase 1: recording only) ===
    struct SpikeRecord {
        uint32_t neuron_id_global;   // global neuron id
        uint64_t timestamp_cycles;   // spike time in cycles
        float v_at_fire;             // membrane potential at firing (pre-reset)
        SpikeRecord(uint32_t gid, uint64_t ts, float v)
            : neuron_id_global(gid), timestamp_cycles(ts), v_at_fire(v) {}
    };

    bool learning_enabled_ = false;          // master switch (default off)
    uint64_t learn_window_cycles_ = 1000;    // window length in cycles
    bool record_membrane_ = false;           // lightweight membrane record guard
    bool record_spike_times_ = true;         // record spike timeline (default on)
    std::string surrogate_type_ = "superspike"; // reserved for Phase 2
    float surrogate_beta_ = 5.0f;                // reserved for Phase 2
    std::string error_file_template_;            // optional error-file per window
    size_t grad_accum_limit_ = 0;                // 0 = unlimited

    // Lightweight buffers for recording
    std::vector<SpikeRecord> spike_history_;
    // Optional: membrane snapshot buffer (kept minimal; filled on spike if enabled)
    std::vector<float> v_mem_history_; // reserved; not filled every cycle to avoid overhead

    // Window tracking (reserved for Phase 2)
    Cycle_t window_start_cycle_ = 0;
    uint64_t current_window_index_ = 0;
    std::vector<float> error_buffer_; // size = num_neurons_

    // Local gradient accumulator keyed to weight address mapping (row-major):
    // key = row*width + col, where row=post_local, col=(pre_global or pre_local based on index mode)
    std::unordered_map<uint64_t, float> local_grad_;

    // Writeback control
    bool apply_writeback_ = false;
    uint32_t apply_every_n_windows_ = 1;
    float learning_rate_ = 0.001f;
    float weight_decay_ = 0.0f;

    // === Helpers (Phase 2) ===
    inline float computeSurrogateGrad_(float v_mem) const {
        // Only superspike for now: 1 / (1 + |beta*(v-vth)|)^2
        const float x = surrogate_beta_ * (v_mem - v_thresh_);
        const float ax = std::fabs(x);
        const float denom = 1.0f + ax * ax;
        return 1.0f / denom;
    }
    std::string replacePlaceholders_(std::string s) const {
        auto repl = [&](const std::string& key, const std::string& val){
            size_t p = 0; while ((p = s.find(key, p)) != std::string::npos) { s.replace(p, key.size(), val); p += val.size(); }
        };
        repl("{node}", std::to_string(node_id_));
        repl("{core}", std::to_string(core_id_));
        repl("{pe}", std::to_string(node_id_)); // alias for node
        return s;
    }
    void onWindowBoundary_(uint64_t window_idx);
    void loadErrorsForWindow_(uint64_t window_idx);
    void applyLocalWeightUpdates_();

    struct NeuronState {
        float v_mem;
        uint32_t refractory_timer;
        Cycle_t last_spike_time;
        NeuronState() : v_mem(0.0f), refractory_timer(0), last_spike_time(0) {}
        NeuronState(float v_r) : v_mem(v_r), refractory_timer(0), last_spike_time(0) {}
    };

    struct PendingMemoryRequest {
        SST::Interfaces::StandardMem::Request::id_t request_id;
        uint64_t address;
        size_t size;
        bool is_row;
        uint32_t pre;
        uint32_t post_start;
        uint32_t count_floats;
        bool has_single_cb;
        uint32_t cb_post;
        std::function<void(float)> single_cb;
    };

    bool clockTick(Cycle_t current_cycle);
    void handleMemoryResponse(SST::Interfaces::StandardMem::Request* req);
    void initializeStatistics();
    void updateNeuronStates();
    void applyLeak(uint32_t neuron_idx);
    void checkAndFireSpike(uint32_t neuron_idx);
    void processLocalSpike(SpikeEvent* spike_event);
    void requestWeight(uint32_t pre_neuron, uint32_t post_neuron, std::function<void(float)> callback);
    bool loadTextWeights(const std::string& weights_file_path);

    SnnPEParentInterface* parent_;
    Output* output_;
    SST::Interfaces::StandardMem* memory_;
    SST::Link* memory_link_;

    int core_id_;
    int total_cores_;
    uint64_t global_neuron_base_;
    uint32_t num_neurons_;
    float v_thresh_;
    float v_reset_;
    float v_rest_;
    float tau_mem_;
    uint32_t t_ref_;
    uint64_t base_addr_;
    uint32_t node_id_;
    int verbose_;
    bool enable_weight_fetch_;
    bool write_weights_on_init_;
    uint64_t memory_warmup_cycles_;
    float init_default_weight_;
    uint32_t max_outstanding_requests_;
    uint32_t max_cache_entries_;
    bool use_event_weight_fallback_;
    bool event_weight_fallback_warned_;
    bool merge_read_cacheline_;
    bool merge_read_row_;
    uint32_t line_size_bytes_;
    // 新增：全网读取模式参数
    uint32_t weights_cols_;   // 列数（例如16x256中的256）
    bool use_post_row_pre_col_; // 索引模式：false=pre_row_post_col，true=post_row_pre_col
    bool enable_detailed_map_log_;
    bool detailed_log_emitted_ = false;

    bool verify_weights_;
    uint32_t weight_verify_samples_;
    float expected_weight_value_;
    float verify_epsilon_;
    bool verify_log_each_sample_;
    bool verify_started_ = false;
    uint32_t verify_requested_ = 0;
    uint32_t verify_completed_ = 0;
    double verify_sum_ = 0.0;
    uint64_t verify_mismatch_count_ = 0;
    bool verify_against_file_ = false;
    std::string verify_file_template_;
    std::vector<float> verify_file_buf_;
    bool verify_file_loaded_ = false;
    
    // 权重文件路径
    std::string weights_file_path_;

    std::vector<NeuronState> neuron_states_;
    // 神经动力学模型（内部策略，不改变外部接口）
    std::unique_ptr<INeuronModel> neuron_model_;
    float dt_ms_ = 1.0f;
    std::queue<SpikeEvent*> incoming_spikes_;
    std::unordered_map<uint64_t, float> weight_cache_;
    std::unordered_map<SST::Interfaces::StandardMem::Request::id_t, PendingMemoryRequest> pending_memory_requests_;
    SST::Interfaces::StandardMem::Request::id_t next_request_id_;
    uint32_t outstanding_requests_ = 0;
    uint32_t pending_reqs_peak_ = 0;

    Cycle_t total_cycles_;
    Cycle_t active_cycles_;
    bool boot_read_sent_;
    bool boot_write_sent_;
    uint32_t delayed_read_counter_;
    bool delayed_read_triggered_ = false;
    bool weights_initialized_;
    bool memory_ready_;
    bool first_cache_hit_logged_ = false;
    bool first_cache_miss_logged_ = false;

    Statistic<uint64_t>* stat_spikes_received_;
    Statistic<uint64_t>* stat_spikes_generated_;
    Statistic<uint64_t>* stat_neurons_fired_;
    Statistic<uint64_t>* stat_memory_requests_;
    Statistic<uint64_t>* stat_weight_cache_hits_;
    Statistic<uint64_t>* stat_weight_cache_misses_;
    Statistic<uint64_t>* stat_merged_reads_rows_;
    Statistic<uint64_t>* stat_merged_reads_cls_;
    Statistic<uint64_t>* stat_weights_verify_count_;
    Statistic<uint64_t>* stat_weights_mismatch_count_;
    Statistic<double>* stat_weights_verify_sum_;
    // 扩展统计
    Statistic<uint64_t>* stat_routes_entries_ = nullptr;
    Statistic<uint64_t>* stat_fanout_per_spike_ = nullptr;
    
    // 内部计数器用于getStatistics()方法
    uint64_t count_spikes_received_;
    uint64_t count_spikes_generated_;
    uint64_t count_neurons_fired_;
    uint64_t count_memory_requests_;

    // ===== 权重驱动路由（可选）=====
    bool routing_weight_driven_ = false;
    std::string weights_template_;
    uint32_t total_nodes_cfg_ = 16;
    float routing_epsilon_ = 1e-8f;
    uint32_t routing_topk_ = 0;
    uint32_t routing_topk_per_pe_ = 0;
    bool route_exclude_self_pe_ = false;
    std::string route_layers_mask_;
    bool route_filter_warn_ = true;
    // 解析后的层间许可表（key=(src_layer<<8)|dst_layer）
    std::unordered_set<uint32_t> allowed_layer_edges_;
    bool allow_all_layers_ = true;
    // 映射框架集成
    std::string mapping_mode_;
    std::string mapping_edges_file_;
    bool mapping_csv_has_header_ = true;
    std::string mapping_csv_separator_ = ",";
    bool mapping_assume_block_ids_ = true;
    // routes_by_source_[pre_global] = list of destination global neuron ids
    // 本地路由表（兼容保留，作为共享表不可用时的回退）
    std::unordered_map<uint32_t, std::vector<uint32_t>> routes_by_source_;
    // 进程范围共享的路由表指针（多个核心/组件共享，避免重复构建与冗余内存）
    std::shared_ptr<const std::unordered_map<uint32_t, std::vector<uint32_t>>> routes_shared_;
    bool buildWeightDrivenRoutes();
    bool buildRoutesFromEdgesCSV();
    uint32_t getLayerIdFromPE(uint32_t pe) const;

    // === 路由共享缓存（进程级）===
    // 说明：使用基于参数集合的key对构建结果做一次性缓存；
    // 命中时各核心直接复用共享表，未命中则首次构建并写回缓存。
    static std::mutex s_route_cache_mtx_;
    using RouteMap = std::unordered_map<uint32_t, std::vector<uint32_t>>;
    static std::unordered_map<std::string, std::weak_ptr<const RouteMap>> s_route_cache_;
    std::string buildRouteCacheKey() const;
};

} // namespace SnnDL
} // namespace SST

#endif // _H_SST_SNN_PE_SUBCOMPONENT

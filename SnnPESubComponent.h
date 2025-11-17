#ifndef _H_SST_SNN_PE_SUBCOMPONENT
#define _H_SST_SNN_PE_SUBCOMPONENT

#include <sst/core/subcomponent.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/shared/sharedArray.h>
#include <queue>
#include <deque>
#include <map>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <functional>
#include <utility>
#include <string>
#include <vector>
#include "SpikeEvent.h"
#include "SnnPEParentInterface.h"
#include "SnnCoreAPI.h"
#include "SnnNeuronModel.h"
#include "SnnProfiler.h"  // 轻量级性能分析（条件编译）

namespace SST {
namespace SnnDL {

class MultiCorePE; // forward declaration for parent cast
class GatherBufferIF;
struct GasOpData;

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
        {"max_cache_entries", "Maximum number of entries in the weight cache", "65536"},
        {"use_event_weight_fallback", "Use weight from spike event if memory fetch fails", "0"},
        {"merge_read_cacheline", "Merge memory reads to cache line size", "1"},
        {"merge_read_row", "Merge memory reads to a full row", "0"},
        {"line_size_bytes", "Cache line size in bytes", "64"},
        {"weights_cols", "Number of columns in weight matrix when using global read (post_row_pre_col)", "0"},
        {"index_mode", "Indexing mode: pre_row_post_col (default) or post_row_pre_col", "pre_row_post_col"},
        {"use_soa_neuron_state", "Use Structure-of-Arrays layout for neuron state (0=AoS,1=SoA)", "0"},
        {"use_aosoa_neuron_state", "Use block-wise AoSoA iteration on top of SoA (0/1)", "0"},
        {"aosoa_block_rows", "AoSoA block row width; defaults to bcsr_block_rows when unset", "0"},
        {"verify_routing_weights", "Log and verify routing fanout against weight threshold (0/1)", "0"},
        {"enable_detailed_map_log", "Enable detailed logging of neuron mapping", "0"},
        {"verify_weights", "Enable weight verification", "0"},
        {"weight_verify_samples", "Number of weight samples to verify", "16"},
        {"expected_weight_value", "Expected weight value for verification", "0.0"},
        {"verify_epsilon", "Epsilon for floating point comparison", "1e-4"},
        {"verify_log_each_sample", "Log each weight sample for verification", "0"},
        {"verify_against_file", "Verify weights by comparing memory reads against weight file (0/1)", "0"},
        {"verify_file_template", "Template for per-PE weight files used for verification (e.g., .../classification_weights_pe_{pe}.bin)", ""},
        {"scatter_diag_limit", "Per-window diagnostic logs for Scatter apply (number of dv>0 entries to print)", "0"},
        {"quiet_finish_logs", "Suppress finish() console summaries (0/1)", "0"},
        {"loader_done_key", "SharedArray key toggled by WeightLoader upon completion", ""},
        // Profiling（可选）
        {"enable_profiler", "Enable lightweight profiling for hot functions (0/1)", "0"},
        {"profiler_csv_prefix", "CSV export prefix for profiler (e.g., outputs_large/<run>/profile_core)", ""},
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
        {"route_filter_warn", "Print prominent warning when route filters are enabled (0/1)", "1"},
        {"readresp_zero_fallback", "When DRAM returns 0 for weight, fallback to init_default_weight (0/1)", "0"},
        // === GAS Apply/Scatter (Phase-1, default off) ===
        {"apply_acc_enable", "Enable Apply-side accumulation and Scatter-side fire (0/1)", "0"},
        {"acc_high_watermark_bytes", "Accumulator high-watermark in bytes before spilling", "16777216"},
        {"acc_spill_enable", "Enable spilling to per-window delta-log when HWM reached (0/1)", "1"},
        {"stage_events_csv", "Optional path to write stage events (seq, begin/end, apply/scatter)", ""},
        // === Online Gating (event-driven, default off) ===
        {"gating_mode", "Gating mode: off|event (default off)", "off"},
        {"gating_ttl_cycles", "TTL window (cycles) for gating decision validity", "1000"},
        {"gating_scope", "Scope of event gating: inputs|all", "inputs"},
        // 映射框架集成（默认关闭）
        {"mapping_mode", "Mapping-driven routes: off (default) or edges_csv", "off"},
        {"mapping_edges_file", "CSV file of edges: src_global,dst_global,weight(optional)", ""},
        {"mapping_csv_has_header", "Edges CSV has header line (0/1)", "1"},
        {"mapping_csv_separator", "Edges CSV separator (default ',')", ","},
        {"mapping_assume_block_ids", "Assume global_id=pe*rows+row for PE inference (0/1)", "1"},
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
        {"weight_decay", "L2 weight decay coefficient (requires cached weight)", "0.0"},
        // === Code-level memory optimizations (optional, default off) ===
        {"use_clock_weight_cache", "Use clock/second-chance policy for weight cache (0/1)", "0"},
        {"apply_dense_acc_enable", "Use dense-array accumulator for GAS Apply (0/1)", "0"},
        // 诊断：禁用权重缓存以强制触发内存读取
        {"disable_weight_cache", "Disable weight cache to force memory reads (diagnostic)", "0"}
        ,
        // 严格GAS：在 BeginApply/Scatter 阶段按窗发起权重读取以填充缓存（不改变ΔV语义，由Scatter统一应用）
        {"window_read_enable", "Issue window-scoped weight reads at BeginApply/Scatter (0/1)", "0"},
        {"window_read_budget", "Max number of (pre,post) single-col reads per window", "1024"},
        // 安全保护：限制单窗边集合容量，防止极端随机发放导致内存增长
        {"edge_collector_max_capacity", "Max edges per window before overflow protection", "1000000"}
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
        {"fanout_per_spike", "Fanout size per emitted spike in weight-driven routing", "count", 1},
        {"cache_evictions", "Number of cache evictions in weight cache (LRU)", "count", 1},
        {"pending_reqs_peak", "Peak number of outstanding memory read requests", "count", 1},
        {"cycles_update_neuron", "Approximate cycles spent updating neuron states (layout-dependent)", "cycles", 1},
        {"synaptic_accesses", "Number of synaptic weight applications during spike processing", "count", 1},
        {"scheme1_bytes_read", "Total bytes read issued by Scheme-1 baseline prefetcher", "bytes", 1},
        // Batch-A detailed memory access stats (can be configured as Histogram in Python)
        {"mem_read_latency_cycles", "End-to-end memory read latency in cycles", "cycles", 1},
        {"mem_read_latency_cycles_weights", "Read latency (cycles) for weight-region accesses", "cycles", 1},
        {"mem_read_latency_cycles_state", "Read latency (cycles) for non-weight (state/other) accesses", "cycles", 1},
        {"mem_req_size_bytes", "Request size in bytes at issue time", "bytes", 1},
        {"mem_outstanding_at_issue", "Outstanding inflight requests at issue time", "count", 1},
        // GAS (windowed gather/apply/scatter): upstream totals accumulated at PE层（由 GatherBufferIF 通过 CustomResp 通知）
        {"gas_unique_reads_total", "Unique coalesced read transactions issued by GAS", "reads", 1},
        {"gas_unique_bytes_total", "Total bytes covered by unique coalesced reads (GAS)", "bytes", 1},
        // Apply/Scatter端到端统计（Phase-1）
        {"gas_apply_acc_updates_total", "Apply阶段的delta累加次数（有效子读）", "count", 1},
        {"gas_acc_posts_touched_total", "Apply阶段触达的post计数（去重）", "posts", 1},
        {"gas_scatter_spikes_emitted_total", "Scatter阶段发放的spike个数", "spikes", 1},
        {"gas_acc_high_watermark_bytes_total", "累加器峰值占用（bytes）", "bytes", 1},
        {"gas_acc_spill_records_total", "溢写到增量日志的记录条数", "records", 1},
        {"gas_acc_spilled_bytes_total", "溢写到增量日志的有效字节数（payload）", "bytes", 1}
        ,
        {"weight_read_requests", "Number of weight read requests issued by core", "requests", 1},
        {"gas_edge_overflow", "Number of times per-window edge collector hit capacity and stopped recording", "events", 1}
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
    void resetMembraneState(float v_rest) override;
    // 回退驱动：当上层无法自动触发clockTick时，由父组件每拍调用一次
    inline void driveOneCycle() { (void)clockTick((Cycle_t)0); }
    // 手动触发：结束当前Gather窗口（仅 manual_window_drive 下有效）
    void forceEndGather() override;

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

    // === GAS Apply/Scatter Phase‑1 ===
    bool apply_acc_enable_ = false;           // gate for end-to-end semantics
    bool readresp_zero_fallback_ = false;     // test-only: map zero read to default weight
    uint64_t acc_hwm_bytes_ = 16 * 1024 * 1024; // default 16MiB
    bool acc_spill_enable_ = true;
    std::string stage_events_csv_;
    enum class GasStage { Idle=0, Gather=1, Apply=2, Scatter=3 };
    GasStage gas_stage_ = GasStage::Idle;
    uint32_t curr_stage_seq_ = 0;             // gather/apply/scatter sequence id
    // GAS superstep timing (ns ~= cycles@1GHz)
    struct StatsReporter {
        SnnPESubComponent* core = nullptr;
        void init(SnnPESubComponent* owner) { core = owner; }
        void reportMemoryIssue(size_t bytes, bool count_weight_read) const;
        void reportApplyScatter(uint64_t acc_updates, uint64_t posts_touched,
                                uint64_t spikes_emitted, uint64_t hwm_bytes,
                                uint64_t spill_records, uint64_t spilled_bytes) const;
        void reportWindowSpikes(uint32_t seq, uint64_t spikes_emitted) const;
        void reportCacheAccess(bool hit) const;
        void updatePendingPeak(uint32_t outstanding) const;
    };

    struct StageEventHub {
        SnnPESubComponent* core = nullptr;
        uint64_t t_begin_gather = 0;
        uint64_t t_begin_apply = 0;
        uint64_t t_begin_scatter = 0;
        bool have_begin_gather = false;
        bool have_begin_apply = false;
        bool have_begin_scatter = false;

        void init(SnnPESubComponent* owner) { core = owner; }
        void markBeginGather(uint32_t seq);
        void markBeginApply(uint32_t seq);
        void markBeginScatter(uint32_t seq);
        void markEndScatter(uint32_t seq, uint64_t spikes_emitted);
    } stage_event_hub_;
    StatsReporter stats_reporter_;
    // Accumulators
    std::unordered_map<uint32_t, float> acc_delta_;        // post_local -> deltaV
    uint64_t acc_bytes_estimate_ = 0;         // approximate memory footprint
    std::vector<std::pair<uint32_t,float>> acc_spill_log_; // fallback when HWM reached
    // Per-window aggregation counters (PE-level flush on stage events)
    uint64_t acc_updates_count_ = 0;
    uint64_t acc_posts_touched_count_ = 0;
    uint64_t acc_spill_records_count_ = 0;
    uint64_t acc_spilled_bytes_sum_ = 0;
    uint64_t acc_hwm_bytes_max_ = 0;
    // Stats pointers
    Statistic<uint64_t>* stat_gas_apply_acc_updates_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_posts_touched_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_scatter_spikes_emitted_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_hwm_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spill_records_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spilled_bytes_total_ = nullptr;
    // GAS superstep duration statistics (cycles)
    Statistic<uint64_t>* stat_gas_superstep_gather_cycles_  = nullptr;
    Statistic<uint64_t>* stat_gas_superstep_apply_cycles_   = nullptr;
    Statistic<uint64_t>* stat_gas_superstep_scatter_cycles_ = nullptr;
    Statistic<uint64_t>* stat_gas_superstep_total_cycles_   = nullptr;
    // Last window spikes emitted (for stage CSV EndScatter)
    uint64_t spikes_emitted_window_ = 0;
    // Accumulate all spikes fired within current GAS window (regardless of phase)
    // 每窗口发放统计（低开销）：
    // - window_spikes_all_: 窗口内由 handleNeuronFire_ 累计（仅当 apply_acc_enable_ && gas_window_mode_）。
    // - spikes_generated_base_: BeginScatter 时记录的累计发放基线；
    //   若 EndScatter 计算到 to_emit==0，则用 (count_spikes_generated_ - spikes_generated_base_) 兜底，
    //   保证窗口口径与总口径对齐，且仅在缺失时触发一次 O(1) 计算。
    uint64_t window_spikes_all_ = 0;
    uint64_t spikes_generated_base_ = 0;
    // Per-window fire gating to ensure at most one spike per neuron per GAS window
    std::vector<uint8_t> fired_this_window_;
    // Per-run unique firing tracker（O(N) 字节/核，统计去重口径）
    std::vector<uint8_t> fired_ever_;
    // Diagnostics: recordEdge outcome counters (per GAS window)
    uint64_t diag_edges_record_hits_ = 0;
    uint64_t diag_edges_stage_skips_ = 0;
    uint64_t diag_edges_cond_skips_ = 0;
    // Diagnostics: spike arrival counters by GAS stage (per window)
    uint64_t diag_spikes_stage_gather_ = 0;
    uint64_t diag_spikes_stage_apply_ = 0;
    uint64_t diag_spikes_stage_scatter_ = 0;
    uint64_t diag_spikes_stage_idle_ = 0;
    struct GasEdgeCollector {
        std::unordered_map<uint64_t, uint32_t> curr;
        std::vector<std::pair<uint64_t, uint32_t>> prev;
        size_t prev_iter_idx = 0;
        bool record_stage_warned = false;
        bool record_cond_warned = false;
        bool capacity_warned = false; // 容量溢出仅告警一次（每窗复位）

        size_t currSize() const { return curr.size(); }
        size_t prevSize() const { return prev.size(); }
        size_t prevIter() const { return prev_iter_idx; }
        bool prevEmpty() const { return prev.empty(); }
        void clearWarnings() { record_stage_warned = record_cond_warned = false; }

        void flipForApply(bool debug, Output* out, int core_id, uint32_t seq) {
            if (debug && out) {
                out->verbose(CALL_INFO, 0, 0,
                    "[diag-edges] BeginApply seq=%u edges_curr=%zu\n",
                    seq, curr.size());
            }
            prev.clear();
            prev.reserve(curr.size());
            for (const auto& kv : curr) {
                prev.emplace_back(kv.first, kv.second);
            }
            if (debug && out) {
                out->verbose(CALL_INFO, 0, 0,
                    "[diag-edges] BeginApply seq=%u edges_prev=%zu\n",
                    seq, prev.size());
            }
            curr.clear();
            prev_iter_idx = 0;
            capacity_warned = false; // 新窗复位
        }

        bool nextPrev(uint64_t& key, uint32_t& count) {
            if (prev_iter_idx >= prev.size()) {
                return false;
            }
            key = prev[prev_iter_idx].first;
            count = prev[prev_iter_idx].second;
            ++prev_iter_idx;
            return true;
        }
    } edge_collector_;
    // Helpers (phase‑1)
    inline void accReset_() {
        acc_spill_log_.clear();
        acc_bytes_estimate_ = 0;
        if (apply_dense_acc_enable_) {
            for (auto idx : acc_touched_list_) {
                if (idx < acc_dense_.size()) {
                    acc_dense_[idx] = 0.0f;
                    if (!acc_touched_bitmap_.empty()) acc_touched_bitmap_[idx] = 0;
                }
            }
            acc_touched_list_.clear();
        } else {
            acc_delta_.clear();
        }
    }
    inline void accUpdate_(uint32_t post, float dv) {
        if (window_read_debug_ && std::fabs(dv) > 1e-6f && debug_edge_log_count_ < 128) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-delta] core=%d post=%u dv=%.6f stage=%d\n",
                core_id_, post, dv, static_cast<int>(gas_stage_));
        }
        // Estimate growth: map entry ~ 16B（粗估）；spill record ~ 8B
        if (acc_spill_enable_ && acc_bytes_estimate_ >= acc_hwm_bytes_) {
            acc_spill_log_.emplace_back(post, dv);
            if (stat_gas_acc_spill_records_total_) stat_gas_acc_spill_records_total_->addData(1);
            if (stat_gas_acc_spilled_bytes_total_) stat_gas_acc_spilled_bytes_total_->addData(sizeof(float));
            // 上卷到PE
            /* PE-level aggregation optional: omitted in header to avoid incomplete type issues */
            acc_spill_records_count_ += 1;
            acc_spilled_bytes_sum_ += sizeof(float);
            return;
        }
        if (apply_dense_acc_enable_) {
            if (post < acc_dense_.size()) {
                acc_dense_[post] += dv;
                if (!acc_touched_bitmap_.empty() && acc_touched_bitmap_[post] == 0) {
                    acc_touched_bitmap_[post] = 1;
                    acc_touched_list_.push_back(post);
                    if (stat_gas_acc_posts_touched_total_) stat_gas_acc_posts_touched_total_->addData(1);
                    acc_posts_touched_count_ += 1;
                    acc_bytes_estimate_ += 16; // rough per new entry
                }
            }
        } else {
            auto it = acc_delta_.find(post);
            if (it == acc_delta_.end()) {
                acc_delta_[post] = dv;
                acc_bytes_estimate_ += 16; // rough
                if (stat_gas_acc_posts_touched_total_) stat_gas_acc_posts_touched_total_->addData(1);
                /* PE-level aggregation optional */
                acc_posts_touched_count_ += 1;
            } else {
                it->second += dv;
            }
        }
        if (stat_gas_apply_acc_updates_total_) stat_gas_apply_acc_updates_total_->addData(1);
        acc_updates_count_ += 1;
        if (stat_gas_acc_hwm_bytes_total_) { stat_gas_acc_hwm_bytes_total_->addData(acc_bytes_estimate_); }
        if (acc_bytes_estimate_ > acc_hwm_bytes_max_) acc_hwm_bytes_max_ = acc_bytes_estimate_;
    }

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

    // === Activity f (per-window active axons ratio) ===
    bool activity_stats_enable_ = true;
    uint32_t activity_window_seq_ = 0;
    std::unordered_set<uint32_t> activity_pre_set_;
    inline void activityReset_() { activity_pre_set_.clear(); }
    inline void recordActivePre_(uint32_t pre_global) {
        if (activity_stats_enable_) activity_pre_set_.insert(pre_global);
    }
    // 容量配置与统计
    size_t edge_collector_max_capacity_ = 1000000;
    bool edge_collector_capacity_warned_ = false;
    Statistic<uint64_t>* stat_gas_edge_overflow_ = nullptr;
    inline MultiCorePE* parentPE_() const { return parent_pe_cached_; }
    inline void recordEdge_(uint32_t post_local, uint32_t pre_global) {
        if (!(enable_weight_fetch_ && memory_ && memory_ready_)) {
            diag_edges_cond_skips_++;
            if (window_read_debug_ && !edge_collector_.record_cond_warned) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-edges] recordEdge skipped enable_weight_fetch=%d memory=%s ready=%d\n",
                    enable_weight_fetch_ ? 1 : 0, memory_ ? "set" : "null", memory_ready_ ? 1 : 0);
                edge_collector_.record_cond_warned = true;
            }
            return;
        }
        bool stage_ok = false;
        switch (gas_stage_) {
            case GasStage::Gather:
                stage_ok = true;
                break;
            case GasStage::Apply:
                stage_ok = record_edge_apply_enable_;
                break;
            case GasStage::Idle:
                stage_ok = record_edge_idle_enable_;
                break;
            case GasStage::Scatter:
                stage_ok = record_edge_scatter_enable_;
                break;
            default:
                stage_ok = false;
                break;
        }
        if (!(apply_acc_enable_ && gas_window_mode_ && stage_ok)) {
            diag_edges_stage_skips_++;
            if (window_read_debug_ && !edge_collector_.record_stage_warned) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-edges] recordEdge skipped apply_acc=%d gas_window=%d stage=%d (apply_en=%d idle_en=%d scatter_en=%d)\n",
                    apply_acc_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0, static_cast<int>(gas_stage_),
                    record_edge_apply_enable_ ? 1 : 0, record_edge_idle_enable_ ? 1 : 0,
                    record_edge_scatter_enable_ ? 1 : 0);
                edge_collector_.record_stage_warned = true;
            }
            return;
        }
        // 容量保护：极端情况下避免map无限增长
        if (edge_collector_.curr.size() >= edge_collector_max_capacity_) {
            if (!edge_collector_.capacity_warned) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-edges] ⚠️ edge_collector capacity reached (core=%d seq=%u cap=%zu), stop recording this window\n",
                    core_id_, curr_stage_seq_, edge_collector_max_capacity_);
                edge_collector_.capacity_warned = true;
                if (stat_gas_edge_overflow_) stat_gas_edge_overflow_->addData(1);
            }
            return;
        }
        uint64_t key = (static_cast<uint64_t>(post_local) << 32) | static_cast<uint64_t>(pre_global);
        edge_collector_.curr[key] += 1;
        diag_edges_record_hits_++;
        if (window_read_debug_ && edge_collector_.currSize() <= 5) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-edges] recordEdge sample core=%d seq=%u post_local=%u pre_global=%u size_now=%zu\n",
                core_id_, curr_stage_seq_, post_local, pre_global, edge_collector_.currSize());
        }
    }
    inline bool isPreLocal_(uint32_t pre_global) const {
        return (pre_global >= global_neuron_base_) &&
               (pre_global < global_neuron_base_ + num_neurons_);
    }
    inline uint32_t remapPreGlobalModulo_(uint32_t pre_global) const {
        if (num_neurons_ == 0) return 0;
        const uint64_t width = static_cast<uint64_t>(num_neurons_);
        const uint64_t base = static_cast<uint64_t>(global_neuron_base_) -
                              static_cast<uint64_t>(static_cast<uint32_t>(core_id_)) * width;
        const uint64_t diff = static_cast<uint64_t>(pre_global) - base;
        return static_cast<uint32_t>(diff % width);
    }
    inline uint32_t mapPreGlobalToLocal_(uint32_t pre_global) const {
        if (isPreLocal_(pre_global)) {
            return static_cast<uint32_t>(pre_global - global_neuron_base_);
        }
        return remapPreGlobalModulo_(pre_global);
    }
    inline uint32_t weightMatrixWidth_() const {
        return use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    }
    struct WeightAccessor {
        WeightAccessor(const SnnPESubComponent& core) : core_(core) {}

        bool resolve(uint32_t pre_global, uint32_t post_local,
                     uint32_t& req_pre, uint32_t& req_post,
                     uint64_t& cache_key, bool allow_remap = false) const {
            if (post_local >= core_.num_neurons_) return false;
            req_post = post_local;
            if (core_.use_post_row_pre_col_) {
                if (pre_global >= core_.weights_cols_) return false;
                cache_key = static_cast<uint64_t>(post_local) * static_cast<uint64_t>(core_.weights_cols_)
                          + static_cast<uint64_t>(pre_global);
                req_pre = pre_global;
                return true;
            }
            uint32_t pre_local = 0;
            if (core_.isPreLocal_(pre_global)) {
                pre_local = static_cast<uint32_t>(pre_global - core_.global_neuron_base_);
            } else {
                if (!allow_remap) return false;
                pre_local = core_.remapPreGlobalModulo_(pre_global);
            }
            cache_key = static_cast<uint64_t>(pre_local) * static_cast<uint64_t>(core_.num_neurons_)
                      + static_cast<uint64_t>(post_local);
            req_pre = pre_local;
            return true;
        }

    private:
        const SnnPESubComponent& core_;
    } weight_accessor_{*this};

    struct DiagSink {
        SnnPESubComponent* core = nullptr;
        void init(SnnPESubComponent* owner) { core = owner; }
        bool enabled() const { return core && core->window_read_debug_ && core->output_; }
        template <typename... Args>
        void log(int, const char* fmt, Args&&... args) const {
            if (!enabled()) return;
            core->output_->verbose(CALL_INFO, 0, 0, fmt, std::forward<Args>(args)...);
        }
    };

    struct ReadOrchestrator {
        SnnPESubComponent* core = nullptr;
        DiagSink diag_;
        void init(SnnPESubComponent* owner) { core = owner; diag_.init(owner); }
        void issueFromEdges();
        void issueFromSets(const std::vector<uint32_t>* posts_to_use,
                           const std::unordered_set<uint32_t>* pres_to_use);
        void issueFromSetsBcsr(const std::vector<uint32_t>* posts_to_use,
                               const std::unordered_set<uint32_t>* pres_to_use);
        void issueFallbackReadsIfNeeded(bool strict_gas_active);
        void logWindowReadSummary_(uint32_t posts_prev, uint32_t pres_prev,
                                   uint32_t posts_curr, uint32_t pres_curr,
                                   bool fallback) const;
        void logFallbackSwitch_() const;
        void logIssuedStats_(uint32_t issued) const;
        void logEdgeFetchStart(size_t prev_edges, uint32_t issued,
                               uint32_t outstanding, uint32_t budget) const;
    private:
        bool canIssueMoreReads_() const;
    };
    ReadOrchestrator read_orchestrator_;
    void activityFlush_();

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
        uint64_t issue_cycle = 0;  // 发起请求时的周期，用于测量往返延迟
        bool is_weight = true;      // 是否属于权重访问（用于区域分组统计）
        bool scheme1_prefetch = false; // 是否属于方案1的预取请求
        // BCSR 扩展
        int bcsr_kind = 0;              // 0=dense, 1=rowptr, 2=colidx, 3=blockdata
        uint32_t bcsr_block_row = 0;
        uint32_t bcsr_target_block_col = 0;
        uint32_t bcsr_intra_row = 0;
        uint32_t bcsr_intra_col = 0;
        uint32_t bcsr_row_start = 0;    // 此行块段起始（全局索引）
        uint32_t bcsr_idx_in_row = 0;   // 目标块在行段内的索引
        uint32_t bcsr_global_block_index = 0; // 目标块的全局索引
        bool bcsr_prefetch_all = false; // 是否作为预取请求
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
    bool loadCSRRowptrFromFile_();
    void logCSRRowptrSummary_();
    void reserveWindowContainers_();
    // Internal helpers for robust memory access
    bool ensureLoaderReady_();
    bool ensureMemoryReady_() const { return memory_ != nullptr && memory_ready_; }
    bool prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                           uint64_t& req_addr, size_t& req_size,
                           bool& is_row, uint32_t& col_start, uint32_t& count_floats) const;
    void issueReadCommon_(uint64_t req_addr, size_t req_size,
                          bool is_row, uint32_t row, uint32_t col_start, uint32_t count_floats,
                          std::function<void(float)> single_cb, uint32_t single_col);

    // 方案1辅助
    inline uint32_t scheme1SliceFromPreGlobal_(uint32_t pre_g) const {
        if (scheme1_slices_ == 0) return 0;
        if (scheme1_partition_mod_) return pre_g % scheme1_slices_;
        uint32_t width = std::max<uint32_t>(1, weights_cols_);
        uint32_t seg = width / scheme1_slices_ + ((width % scheme1_slices_) ? 1 : 0);
        uint32_t idx = pre_g / std::max<uint32_t>(1, seg);
        return (idx >= scheme1_slices_) ? (scheme1_slices_ - 1) : idx;
    }
    void scheme1Reset_();
    bool scheme1Tick_(); // 返回 true 表示本周期已由方案1路径完全处理
    void scheme1PrefetchSlice_(uint32_t slice_idx);

    // Debug/diagnostic switches (params)
    bool read_force_single_ = false; // 当为真时，强制按单元素读取（req_size=4B），用于定位对齐/切片问题

    SnnPEParentInterface* parent_;
    MultiCorePE* parent_pe_cached_ = nullptr;
    Output* output_;
    SST::Interfaces::StandardMem* memory_;
    GatherBufferIF* gather_buffer_if_ = nullptr;
    bool manual_window_tick_logged_ = false;
    bool clock_tick_logged_ = false;
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
    bool merge_read_auto_ = false; // auto choose best merge strategy (default off)
    uint32_t line_size_bytes_;
    // GAS control (component-driven phases)
    bool gas_enable_ = false; // enable GAS control-plane (v1: Begin/EndGather per tick)
    bool gas_window_mode_ = false; // 当为true时，不再每周期发送Begin/EndGather，由下游window驱动
    bool gas_manual_window_drive_ = false;
    uint64_t manual_gas_counter_ = 0;
    uint64_t manual_gas_gather_cycles_cfg_ = 200; // fallback
    bool manual_tick_sampled_ = false;
    std::string loader_done_key_;
    bool wait_for_loader_done_ = false;
    bool loader_ready_latched_ = false;
    bool loader_ready_logged_ = false;
    SST::Shared::SharedArray<int> loader_done_shared_;
    bool loader_done_shared_initialized_ = false;

    // ===== 方案1（slice 顺序执行）开关与参数 =====
    // 说明：在一个 superstep 内，先 Gather 收集所有脉冲事件，然后按 slice=0..N-1 的顺序：
    //  1) 载入该 slice 需要的数据（预取对应的权重区间到本地缓存）
    //  2) 对该 slice 的所有脉冲执行计算（Apply）
    //  3) 在 Scatter 阶段统一触发膜电位发放与外发（可选）
    // 默认关闭；启用后将覆盖常规逐周期处理路径。
    bool scheme1_enable_ = false;
    uint32_t scheme1_slices_ = 8;              // 切片数
    uint64_t scheme1_gather_cycles_cfg_ = 100; // Gather 窗口时长（cycles）
    uint64_t scheme1_slice_gap_cycles_ = 0;    // 相邻 slice 间的间隔（cycles）
    uint64_t scheme1_scatter_cycles_ = 1;      // Scatter 持续时长（cycles）
    // 分片映射方式：按 pre_global 连续区间划分（更贴近行/列局部性）
    bool scheme1_partition_mod_ = false;       // 为 true 则按取模；默认按连续区间
    // 运行期状态
    enum class Scheme1Stage { Idle=0, Gather=1, Apply=2, Scatter=3 };
    Scheme1Stage scheme1_stage_ = Scheme1Stage::Idle;
    uint32_t scheme1_current_slice_ = 0;
    uint64_t scheme1_stage_counter_ = 0;
    bool scheme1_prefetch_issued_ = false;
    uint32_t scheme1_pending_prefetch_ = 0;    // 仍在等待的预取响应计数
    bool s1_is_issuing_prefetch_ = false;      // 给 issueReadCommon_ 打标签用
    std::vector<std::deque<SpikeEvent*>> scheme1_slice_queues_;
    bool scheme1_queues_inited_ = false;       // 仅首次分配队列；之后跨superstep保留
    bool scheme1_first_superstep_ = true;      // 第一次superstep用于初始化current_slice_

    // 新增：全网读取模式参数
    uint32_t weights_cols_;   // 列数（例如16x256中的256）
    bool use_post_row_pre_col_; // 索引模式：false=pre_row_post_col，true=post_row_pre_col
    bool enable_detailed_map_log_;
    bool log_weight_details_;
    bool detailed_log_emitted_ = false;
    bool verify_routing_weights_ = false;

    bool verify_weights_;
    uint32_t weight_verify_samples_;
    float expected_weight_value_;
    float verify_epsilon_;
    bool verify_log_each_sample_;
    // Optional barrier cycles to allow external weight loaders to finish (default 0)
    uint64_t loader_barrier_cycles_ = 0;
    bool verify_started_ = false;
    uint32_t verify_requested_ = 0;
    uint32_t verify_completed_ = 0;
    double verify_sum_ = 0.0;
    uint64_t verify_mismatch_count_ = 0;
    bool verify_against_file_ = false;
    std::string verify_file_template_;
    std::vector<float> verify_file_buf_;
    bool verify_file_loaded_ = false;
    bool verify_cluster_enable_ = false; // 将前若干样本聚类到同一cacheline，便于观测L2命中（默认关闭）
    // 控制收尾阶段的控制台日志
    bool quiet_finish_logs_ = false;
    
    // 权重文件路径
    std::string weights_file_path_;

    std::vector<NeuronState> neuron_states_;
    bool use_soa_state_ = false;
    bool use_aosoa_state_ = false;
    alignas(64) std::vector<float> soa_v_mem_;
    alignas(64) std::vector<uint32_t> soa_refrac_;
    alignas(64) std::vector<Cycle_t> soa_last_spike_;
    uint32_t aosoa_block_rows_ = 16;
    inline float getMem_(uint32_t idx) const {
        return use_soa_state_ ? soa_v_mem_[idx] : neuron_states_[idx].v_mem;
    }
    inline void setMem_(uint32_t idx, float val) {
        if (use_soa_state_) soa_v_mem_[idx] = val;
        else neuron_states_[idx].v_mem = val;
    }
    inline uint32_t getRefrac_(uint32_t idx) const {
        return use_soa_state_ ? soa_refrac_[idx] : neuron_states_[idx].refractory_timer;
    }
    inline void setRefrac_(uint32_t idx, uint32_t val) {
        if (use_soa_state_) soa_refrac_[idx] = val;
        else neuron_states_[idx].refractory_timer = val;
    }
    inline Cycle_t getLastSpike_(uint32_t idx) const {
        return use_soa_state_ ? soa_last_spike_[idx] : neuron_states_[idx].last_spike_time;
    }
    inline void setLastSpike_(uint32_t idx, Cycle_t val) {
        if (use_soa_state_) soa_last_spike_[idx] = val;
        else neuron_states_[idx].last_spike_time = val;
    }
    inline void recordSynapticAccess_() {
        count_synaptic_accesses_++;
        if (stat_synaptic_accesses_) stat_synaptic_accesses_->addData(1);
    }
    void updateNeuronStatesAoS_();
    void updateNeuronStatesSoA_();
    void updateNeuronStatesAoSoA_();
    void checkAndFireSpikeAoS_(uint32_t neuron_idx);
    void checkAndFireSpikeSoA_(uint32_t neuron_idx);
    void checkAndFireSpikeAoSoA_(uint32_t neuron_idx);
    void handleNeuronFire_(uint32_t neuron_idx, float v_before, float v_after);
    // 神经动力学模型（内部策略，不改变外部接口）
    std::unique_ptr<INeuronModel> neuron_model_;
    float dt_ms_ = 1.0f;
    std::queue<SpikeEvent*> incoming_spikes_;
    // LRU weight cache: key = linear index, value = float weight
    struct CacheEntry { float value; std::list<uint64_t>::iterator it; };
    std::list<uint64_t> cache_lru_list_;               // MRU at front, LRU at back
    std::unordered_map<uint64_t, CacheEntry> weight_cache_;
    // Clock weight cache（可选）
    bool use_clock_weight_cache_ = false;
    bool disable_weight_cache_ = false; // 诊断用：true时cacheGet_总返回false
    inline bool weightCacheTryGet_(uint64_t key, float& out) {
        return use_clock_weight_cache_ ? clockGet_(key, out) : cacheGet_(key, out);
    }
    inline void weightCacheStore_(uint64_t key, float value) {
        if (use_clock_weight_cache_) {
            clockPut_(key, value);
        } else {
            cachePut_(key, value);
        }
    }
    bool window_read_enable_ = false;   // 严格GAS：按窗发起权重读取
    uint32_t window_read_budget_ = 1024;
    bool window_read_debug_ = false;    // 控制窗口读相关调试日志
    // 每窗Scatter阶段诊断配额（打印dv/v调试信息的条数，避免日志爆炸）
    uint32_t scatter_diag_limit_param_ = 0;
    uint32_t scatter_diag_quota_ = 0;
    std::vector<uint8_t> posts_seen_window_;      // 当前窗口触达的 post（Scatter 填充，位图）
    std::vector<uint8_t> posts_seen_prev_window_; // 上一窗口触达的 post（位图，给 BeginApply 使用）
    std::vector<uint32_t> posts_list_window_;     // 当前窗口触达的 post 列表（压缩遍历）
    std::vector<uint32_t> posts_list_prev_window_;// 上一窗口触达的 post 列表（压缩遍历）
    std::unordered_set<uint32_t> active_pre_window_;      // 当前窗口触达的 pre（全局列）
    std::unordered_set<uint32_t> active_pre_prev_window_; // 上一窗口触达的 pre
    uint32_t debug_window_log_count_ = 0;
    uint32_t debug_edge_log_count_ = 0;
    // Debug instrumentation
    uint32_t debug_window_idx_ = 0;
    std::vector<uint64_t> wcache_keys_;
    std::vector<float>    wcache_vals_;
    std::vector<uint8_t>  wcache_access_;
    std::unordered_map<uint64_t, uint32_t> wcache_index_;
    uint32_t wcache_hand_ = 0;
    uint32_t wcache_size_ = 0;
    uint32_t wcache_cap_  = 0;
    // LRU helpers (internal)
    bool cacheGet_(uint64_t key, float& out);
    void cachePut_(uint64_t key, float value);
    bool clockGet_(uint64_t key, float& out);
    void clockPut_(uint64_t key, float value);
    std::unordered_map<SST::Interfaces::StandardMem::Request::id_t, PendingMemoryRequest> pending_memory_requests_;
    SST::Interfaces::StandardMem::Request::id_t next_request_id_;
    uint32_t outstanding_requests_ = 0;
    uint32_t pending_reqs_peak_ = 0;
    uint32_t window_reads_issued_this_apply_ = 0; // 本 Apply 窗口内已发起的读取数（用于预算）
    uint64_t bcsr_req_edges_ = 0;
    uint64_t bcsr_req_wait_rowptr_ = 0;
    uint64_t bcsr_req_block_hit_ = 0;
    uint64_t bcsr_req_block_miss_ = 0;
    // Dense Apply accumulator（可选）
    bool apply_dense_acc_enable_ = false;
    std::vector<float>   acc_dense_;
    std::vector<uint8_t> acc_touched_bitmap_;
    std::vector<uint32_t> acc_touched_list_;

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
    Statistic<uint64_t>* stat_cache_evictions_ = nullptr;
    Statistic<uint64_t>* stat_pending_reqs_peak_ = nullptr;
    Statistic<uint64_t>* stat_cycles_update_neuron_ = nullptr;
    Statistic<uint64_t>* stat_synaptic_accesses_ = nullptr;
    // Scheme1 baseline bytes read (sum of issued Read sizes in scheme1PrefetchSlice_)
    Statistic<uint64_t>* stat_s1_bytes_read_ = nullptr;
    // 门控诊断：权重读请求发起计数（用于判定发起端是否触发）
    Statistic<uint64_t>* stat_weight_read_requests_ = nullptr;
    Statistic<uint64_t>* stat_window_reads_issued_total_ = nullptr;
    // GAS totals accumulated from GatherBufferIF via CustomResp
    Statistic<uint64_t>* stat_gas_unique_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_unique_bytes_total_ = nullptr;
    
    // 内部计数器用于getStatistics()方法
    uint64_t count_spikes_received_;
    uint64_t count_spikes_generated_;
    uint64_t count_neurons_fired_;
    uint64_t count_memory_requests_;
    // 内部计数：用于收尾摘要打印（不依赖SST统计聚合）
    uint64_t count_cache_hits_ = 0;
    uint64_t count_cache_misses_ = 0;
    uint64_t count_merged_reads_rows_ = 0;
    uint64_t count_merged_reads_cls_ = 0;
    uint64_t count_cache_evictions_ = 0;
    // 往返延迟测量（Cycle）
    uint64_t accum_mem_latency_cycles_ = 0;
    uint64_t count_mem_responses_ = 0;
    uint64_t count_cycles_update_neuron_ = 0;
    uint64_t count_synaptic_accesses_ = 0;

    // === Batch-A: Memory access detailed statistics (per-core) ===
    // Histogram candidates (type configured via Python):
    //  - mem_read_latency_cycles: 端到端读延迟
    //  - mem_read_latency_cycles_weights/state: 区域分组
    //  - mem_req_size_bytes: 请求字节大小
    //  - mem_outstanding_at_issue: 发起时并发请求数
    Statistic<uint64_t>* stat_mem_read_latency_cycles_ = nullptr;
    Statistic<uint64_t>* stat_mem_read_latency_cycles_weights_ = nullptr;
    Statistic<uint64_t>* stat_mem_read_latency_cycles_state_ = nullptr;
    Statistic<uint64_t>* stat_mem_req_size_bytes_ = nullptr;
    Statistic<uint64_t>* stat_mem_outstanding_at_issue_ = nullptr;

    // Dense 权重区域上界（用于区域分组）；BCSR 通过 bcsr_kind 判别
    uint64_t weight_region_end_ = 0; // [base_addr_, weight_region_end_) 视为权重区（dense）

    // ===== BCSR 读路径支持 =====
    bool use_bcsr_ = false;
    uint32_t bcsr_br_ = 16;                 // block rows
    uint32_t bcsr_bc_ = 16;                 // block cols
    uint32_t bcsr_val_bytes_ = 4;           // FP32
    uint32_t bcsr_idx_bytes_ = 2;           // uint16
    uint64_t bcsr_rowptr_addr_ = 0;         // rowptr 基地址（全局）
    uint64_t bcsr_colidx_addr_ = 0;         // colidx 基地址
    uint64_t bcsr_blockdata_addr_ = 0;      // blockdata 基地址
    uint64_t bcsr_blockids_addr_ = 0;       // blockids 基地址（可选）
    // 读值守护（诊断/健壮性）：过滤非有限或异常大的权重，避免毒化ΔV（默认开启）
    bool bcsr_weight_guard_enable_ = true;
    float bcsr_weight_abs_max_ = 10.0f;
    uint64_t bcsr_bad_weight_count_ = 0;
    std::vector<uint32_t> bcsr_rowptr_host_;// 缓存 rowptr 表（内存镜像）
    bool bcsr_rowptr_ready_ = false;

    // --- Verify (BCSR probe) state (diagnostic only; zero-cost when disabled) ---
    bool verify_bcsr_started_ = false;
    bool verify_bcsr_done_ = false;
    bool verify_bcsr_inflight_ = false;
    uint32_t verify_bcsr_post_local_ = 0;   // target row (local)
    uint32_t verify_bcsr_block_col_ = 0;    // chosen block column
    uint32_t verify_bcsr_intra_col_ = 0;    // scan 0..bc-1
    bool verify_bcsr_block_resolved_ = false;
    uint64_t bcsr_rowptr_retry_cycle_ = 0;
    uint64_t bcsr_rowptr_retry_interval_cycles_ = 2000;
    uint32_t bcsr_rowptr_retry_attempts_ = 0;
    uint32_t bcsr_rowptr_retry_max_ = 4;
    uint32_t bcsr_row_index_cache_cap_ = 64; // 行索引段缓存容量（行数）
    uint32_t bcsr_block_cache_cap_ = 256;    // 数据块缓存容量（块数）
    // 行索引缓存：block_row -> colidx 段
    std::unordered_map<uint32_t, std::vector<uint32_t>> bcsr_row_index_cache_;
    std::list<uint32_t> bcsr_row_index_lru_;
    // 数据块缓存：key=(block_row<<32)|block_col -> 块数据
    struct BcsrBlockEntry { std::vector<float> data; std::list<uint64_t>::iterator it; };
    std::unordered_map<uint64_t, BcsrBlockEntry> bcsr_block_cache_;
    std::list<uint64_t> bcsr_block_lru_;

    // BCSR 统计计数（收尾打印）
    uint64_t bcsr_count_row_reads_ = 0;
    uint64_t bcsr_count_colidx_reads_ = 0;
    uint64_t bcsr_count_block_reads_ = 0;
    uint64_t bcsr_count_block_hits_ = 0;
    uint64_t bcsr_count_block_misses_ = 0;
    uint64_t bcsr_count_row_index_hits_ = 0;
    uint64_t bcsr_count_row_index_fills_ = 0;
    uint64_t bcsr_bytes_idx_ = 0;
    uint64_t bcsr_bytes_val_ = 0;
    bool bcsr_prefetch_all_ = true;
    bool bcsr_prefetch_issued_ = false;

    // 强制探针（仅诊断，PE0/core0用，避免刷屏每窗仅一次）
    uint32_t verify_forced_seq_ = UINT32_MAX;

    // BCSR 辅助
    void requestWeightBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb);
    bool bcsrRowIndexGet_(uint32_t block_row, std::vector<uint32_t>& out);
    void bcsrRowIndexPut_(uint32_t block_row, std::vector<uint32_t>& data);
    bool bcsrBlockGet_(uint32_t block_row, uint32_t block_col, std::vector<float>& out);
    void bcsrBlockPut_(uint32_t block_row, uint32_t block_col, std::vector<float>& data);
    void bcsrPrefetchAll_();
    void bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start);
    void bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk);
    bool loadBcsrRowptrFromFile_();
    bool bcsr_force_file_read_ = false; // 诊断：强制从文件读取BCSR块（绕过内存）
    // 从权重文件读取单个 (post_local, pre_global) 的权重（仅诊断/验证用）
    float readBcsrWeightFromFile_(uint32_t post_local, uint32_t pre_global) const;

    // CSR 已移除

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
    bool record_edge_apply_enable_ = false;
    bool record_edge_idle_enable_ = true;
    bool record_edge_scatter_enable_ = false;
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
    bool routing_diag_logged_ = false;
    bool buildWeightDrivenRoutes();
    bool buildRoutesFromEdgesCSV();
    bool buildWeightDrivenRoutesFromBcsr();
    bool appendRoutesFromBcsrFile(const std::string& path, uint32_t pe_index, int core_index, uint32_t rows_hint);
    bool parseBcsrMeta(const std::string& meta_path, uint32_t& rows_out, uint32_t& cols_out,
                       uint32_t& br_out, uint32_t& bc_out,
                       uint32_t& idx_bytes_out, uint32_t& val_bytes_out,
                       uint64_t& rowptr_off_out, uint64_t& colidx_off_out,
                       uint64_t& blockdata_off_out, uint64_t& blockids_off_out,
                       uint32_t& total_blocks_out) const;
    std::string resolveWeightTemplate(uint32_t pe, int core) const;
    uint32_t getLayerIdFromPE(uint32_t pe) const;
    // Helpers to reduce duplication in route building
    bool applyRouteFilter(uint32_t src_global, uint32_t dst_global, uint32_t rows) const;
    // Helper: given candidates pre_global -> [(score, dst_global)],
    // apply per-PE topk and/or global topk, then fill routes_by_source_.
    // When group_by_pe=false, all dst are treated as in one group (used when we
    // cannot infer PE id from dst_global reliably).
    void buildRoutesFromCandidates(
        const std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>>& tmp,
        uint32_t rows,
        bool group_by_pe);

    // === 路由共享缓存（进程级）===
    // 说明：使用基于参数集合的key对构建结果做一次性缓存；
    // 命中时各核心直接复用共享表，未命中则首次构建并写回缓存。
    static std::mutex s_route_cache_mtx_;
    using RouteMap = std::unordered_map<uint32_t, std::vector<uint32_t>>;
    static std::unordered_map<std::string, std::weak_ptr<const RouteMap>> s_route_cache_;
    std::string buildRouteCacheKey() const;
    void logRoutingSummary_(const char* phase, const char* reason = nullptr);
    // Stage 事件 CSV 写出互斥，避免多核心重复写表头
    static std::mutex s_stage_csv_mutex_;
    static std::unordered_set<std::string> s_stage_csv_files_;
    void appendStageEventRow_(const char* event_name, uint64_t now_ns, uint64_t spikes_emitted);
    void handleStageEventWithoutApply_(const GasOpData* op);
    void prepareEdgeWindowForApply_();
    void diagEdgeWeight_(const char* tag, uint32_t post_local, uint32_t pre_global,
                         float weight, uint32_t count);
    void issueEdgeWeightFetches_();
    void logBcsrWindowStats_(const char* tag);
    void resetBcsrWindowCounters_();

    // ===== 门控事件缓存 =====
    struct GatingEntry { std::vector<uint32_t> dest_pes; uint64_t expire_cycle; };
    bool gating_event_mode_ = false;
    uint64_t gating_ttl_cycles_cfg_ = 1000;
    bool gating_scope_inputs_only_ = true;
    std::unordered_map<uint32_t, GatingEntry> gating_cache_; // key=src_global

public:
    // 应用门控决策（由父级MultiCorePE调用）
    void applyGatingDecision(uint32_t src_global, const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle, uint64_t ttl_cycles);
};

} // namespace SnnDL
} // namespace SST

#endif // _H_SST_SNN_PE_SUBCOMPONENT

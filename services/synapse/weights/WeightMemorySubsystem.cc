// -*- c++ -*-
//
// WeightMemorySubsystem.cc:
// Phase E: 将 StandardMem pending/回调/解析/BCSR 缓存等数据路径收敛到内存子系统，
// 控制层仅保留 GAS/窗口编排与统计汇总。

#include <sst/core/sst_config.h>

#include "WeightMemorySubsystem.h"

#include "SnnBcsrWeightManager.h"

#include <sst/core/output.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace SST;
using namespace SST::SnnDL;

namespace {
static inline bool isPow2(uint32_t v) { return v != 0 && (v & (v - 1u)) == 0; }
static inline uint64_t alignDownU64(uint64_t v, uint64_t a) { return a ? (v & ~(a - 1u)) : v; }
static inline uint64_t alignUpU64(uint64_t v, uint64_t a) {
    if (!a) return v;
    const uint64_t m = a - 1u;
    return (v + m) & ~m;
}

static inline void prepareAlignedRead(uint64_t addr, size_t size, uint32_t line_size,
                                      uint64_t& out_addr, size_t& out_size, size_t& out_slice_off) {
    out_addr = addr;
    out_size = size;
    out_slice_off = 0;
    if (size == 0) return;
    if (!isPow2(line_size)) return;
    const uint64_t ls = static_cast<uint64_t>(line_size);
    const uint64_t end = addr + static_cast<uint64_t>(size);
    const uint64_t a0 = alignDownU64(addr, ls);
    const uint64_t a1 = alignUpU64(end, ls);
    if (a1 <= a0) return;
    out_addr = a0;
    out_size = static_cast<size_t>(a1 - a0);
    out_slice_off = static_cast<size_t>(addr - a0);
}
} // namespace

void WeightMemorySubsystem::onClockTick(uint64_t now_cycle) {
    setNowCycle(now_cycle);
    // BCSR rowptr 预取与后续 prefetchAll 都在内存子系统内闭环
    maybeIssueBcsrRowptrPrefetch_();
}

uint64_t WeightMemorySubsystem::issueRead_(PendingMeta meta) {
    if (!mem_access_ || meta.size == 0) {
        if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
        return 0;
    }
    if (orch_.report_mem_issue) {
        orch_.report_mem_issue(meta.size, meta.count_weight_read);
    }
    const uint64_t addr = meta.address;
    const size_t bytes = meta.size;
    return mem_access_->read(
        addr, bytes,
        [this, meta = std::move(meta)](uint64_t req_id, uint64_t resp_addr, std::vector<uint8_t>&& data) mutable {
            handleReadResp_(req_id, resp_addr, std::move(meta), std::move(data));
        });
}

bool WeightMemorySubsystem::prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                                             uint64_t& req_addr, size_t& req_size,
                                             bool& is_row, uint32_t& col_start, uint32_t& count_floats) const {
    const uint32_t bpf = sizeof(float);
    req_addr = orch_.base_addr + (static_cast<uint64_t>(row) * static_cast<uint64_t>(width) + col) * bpf;
    req_size = sizeof(float);
    is_row = false;
    col_start = col;
    count_floats = 1;
    if (orch_.read_force_single) return true;

    const bool merge_row = orch_.merge_read_row;
    const bool merge_cl = orch_.merge_read_cacheline;
    const bool merge_auto = orch_.merge_read_auto;

    if (merge_auto) {
        const uint32_t fpl = std::max<uint32_t>(1, orch_.line_size_bytes / bpf);
        const size_t bytes_row = static_cast<size_t>(width) * static_cast<size_t>(bpf);
        const size_t bytes_cl = static_cast<size_t>(fpl) * static_cast<size_t>(bpf);
        const bool choose_row = merge_row && (bytes_row <= bytes_cl);
        if (choose_row) {
            is_row = true;
            col_start = 0;
            count_floats = width;
            req_addr = orch_.base_addr + static_cast<uint64_t>(row) * static_cast<uint64_t>(width) * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        } else if (merge_cl) {
            col_start = (col / fpl) * fpl;
            count_floats = std::min<uint32_t>(fpl, width - col_start);
            req_addr = orch_.base_addr +
                       (static_cast<uint64_t>(row) * static_cast<uint64_t>(width) + col_start) * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        }
        return true;
    }

    if (merge_row) {
        is_row = true;
        col_start = 0;
        count_floats = width;
        req_addr = orch_.base_addr + static_cast<uint64_t>(row) * static_cast<uint64_t>(width) * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
        return true;
    }

    if (merge_cl) {
        const uint32_t fpl = std::max<uint32_t>(1, orch_.line_size_bytes / bpf);
        col_start = (col / fpl) * fpl;
        count_floats = std::min<uint32_t>(fpl, width - col_start);
        req_addr = orch_.base_addr +
                   (static_cast<uint64_t>(row) * static_cast<uint64_t>(width) + col_start) * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
        return true;
    }

    return true;
}

void WeightMemorySubsystem::issueDenseResolved_(uint32_t row, uint32_t col, uint32_t cb_col,
                                               std::function<void(float)> cb) {
    if (!mem_access_) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    const uint32_t width = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
    if (width == 0) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }

    uint64_t req_addr = 0;
    size_t req_size = sizeof(float);
    bool is_row = false;
    uint32_t col_start = col;
    uint32_t count_floats = 1;
    (void)prepareDenseRead_(row, col, width, req_addr, req_size, is_row, col_start, count_floats);

    PendingMeta meta{};
    meta.address = req_addr;
    meta.size = req_size;
    meta.is_row = is_row;
    meta.pre = row;
    meta.post_start = col_start;
    meta.count_floats = count_floats;
    meta.has_single_cb = (cb != nullptr);
    meta.cb_post = cb_col;
    meta.single_cb = std::move(cb);
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 0;
    meta.is_weight = (req_addr >= orch_.base_addr && req_addr < orch_.weight_region_end);
    meta.count_weight_read = true;
    (void)issueRead_(std::move(meta));
}

bool WeightMemorySubsystem::issueDensePrefetchRaw(uint64_t req_addr, size_t req_size,
                                                  uint32_t row, uint32_t col_start, uint32_t count_floats,
                                                  bool scheme1_prefetch) {
    if (!mem_access_) return false;
    if (req_size == 0) return false;
    PendingMeta meta{};
    meta.address = req_addr;
    meta.size = req_size;
    meta.is_row = false;
    meta.pre = row;
    meta.post_start = col_start;
    meta.count_floats = count_floats;
    meta.has_single_cb = false;
    meta.cb_post = 0;
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 0;
    meta.is_weight = (req_addr >= orch_.base_addr && req_addr < orch_.weight_region_end);
    meta.count_weight_read = true;
    meta.scheme1_prefetch = scheme1_prefetch;
    return issueRead_(std::move(meta)) != 0;
}

void WeightMemorySubsystem::requestDense_(uint32_t pre, uint32_t post, std::function<void(float)> cb) {
    // 兼容：当启用 BCSR 且仍通过 Dense 接口发起时，自动走 BCSR。
    if (orch_.use_bcsr && orch_.use_post_row_pre_col) {
        requestBCSR_(pre, post, std::move(cb));
        return;
    }
    if (!orch_.accessor) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    uint32_t req_pre = 0;
    uint32_t req_post = 0;
    uint64_t cache_key = 0;
    if (!orch_.accessor->resolve(pre, post, req_pre, req_post, cache_key)) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    const uint32_t row_idx = orch_.use_post_row_pre_col ? req_post : req_pre;
    const uint32_t col_idx = orch_.use_post_row_pre_col ? req_pre : req_post;
    issueDenseResolved_(row_idx, col_idx, /*cb_col*/col_idx, std::move(cb));
}

static float applyWeightGuards(const WeightMemorySubsystem::OrchestratorConfig& orch, float w) {
    if (orch.bcsr_weight_guard_enable) {
        if (!std::isfinite(w) || std::fabs(w) > orch.bcsr_weight_abs_max) {
            return 0.0f;
        }
    }
    if (orch.readresp_zero_fallback && w == 0.0f) return orch.init_default_weight;
    return w;
}

void WeightMemorySubsystem::requestBCSR_(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) {
        if (cb) cb(0.0f);
        return;
    }
    if (!mem_access_) {
        if (cb) cb(0.0f);
        return;
    }
    if (orch_.bcsr_force_file_read && orch_.read_bcsr_from_file) {
        float w = orch_.read_bcsr_from_file(post_local, pre_global);
        if (cb) cb(applyWeightGuards(orch_, w));
        return;
    }
    if (orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
        if (cb) cb(0.0f);
        return;
    }

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
    const uint32_t block_row = (br ? (post_local / br) : 0);
    const uint32_t intra_row = (br ? (post_local % br) : 0);
    const uint32_t block_col = (bc ? (pre_global / bc) : 0);
    const uint32_t intra_col = (bc ? (pre_global % bc) : 0);

    uint32_t start = 0;
    uint32_t end = 0;
    if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) {
        if (cb) cb(0.0f);
        return;
    }
    if (end <= start) {
        if (cb) cb(0.0f);
        return;
    }

    std::vector<uint32_t> cols;
    if (!orch_.bcsr_mgr->rowIndexGet(block_row, cols)) {
        const uint32_t block_count = end - start;
        const size_t bytes = orch_.bcsr_mgr->colIndexBytes(block_count);
        const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(start);
        // NOTE: 对 colidx 段不要扩展到整 cacheline 读（例如 450B→512B）。在 memHierarchy.Cache
        // 的 non-coherent/L1 配置下，多 cacheline 的“扩展读”在部分地址上会返回全 0 的 rr->data，
        // 而同地址的 64B 单行读却是正确的；这会直接导致 BCSR miss→权重=0→发放归零。
        // 保持与稳定版本一致：按原始 addr/bytes 发起。
        const uint64_t req_addr = addr;
        const size_t req_size = bytes;
        const size_t slice_off = 0;
        PendingMeta meta{};
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = bytes;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 2;
        meta.bcsr_block_row = block_row;
        meta.bcsr_target_block_col = block_col;
        meta.bcsr_intra_row = intra_row;
        meta.bcsr_intra_col = intra_col;
        meta.bcsr_row_start = start;
        meta.has_single_cb = (cb != nullptr);
        meta.single_cb = std::move(cb);
        meta.is_weight = true; // 归类为权重域（延迟分组）
        meta.count_weight_read = false;
        // 诊断：colidx 读取发起地址/跨度
        static int dbg_colidx_issue = 0;
        if (diag_out_ && diag_node_id_ == 0 && dbg_colidx_issue < 64) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr-colidx-issue] node=%d core=%d pre=%u post=%u block_row=%u start=%u end=%u addr=0x%llx bytes=%zu aligned=(0x%llx,%zu off=%zu) target_block_col=%u intra=(%u,%u)\n",
                diag_node_id_, diag_core_id_, pre_global, post_local, block_row, start, end,
                (unsigned long long)addr, bytes,
                (unsigned long long)req_addr, req_size, slice_off,
                block_col, intra_row, intra_col);
            ++dbg_colidx_issue;
        }
        (void)issueRead_(std::move(meta));
        return;
    }

    uint32_t idx_in_row = 0;
    bool found = false;
    for (size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == block_col) {
            idx_in_row = static_cast<uint32_t>(i);
            found = true;
            break;
        }
    }
    if (!found) {
        if (cb) cb(0.0f);
        return;
    }

    std::vector<float> blk;
    if (orch_.bcsr_mgr->blockGet(block_row, block_col, blk)) {
        const uint32_t off = intra_row * bc + intra_col;
        float w = (off < blk.size()) ? blk[off] : 0.0f;
        if (cb) cb(applyWeightGuards(orch_, w));
        return;
    }

    const uint32_t global_block_index = start + idx_in_row;
    const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
    const uint64_t addr = orch_.bcsr_mgr->blockDataAddr(global_block_index);
    uint64_t req_addr = addr;
    size_t req_size = block_bytes;
    size_t slice_off = 0;
    prepareAlignedRead(addr, block_bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);
    PendingMeta meta{};
    meta.address = req_addr;
    meta.size = req_size;
    meta.orig_address = addr;
    meta.orig_size = block_bytes;
    meta.slice_offset = slice_off;
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 3;
    meta.bcsr_block_row = block_row;
    meta.bcsr_target_block_col = block_col;
    meta.bcsr_intra_row = intra_row;
    meta.bcsr_intra_col = intra_col;
    meta.bcsr_row_start = start;
    meta.bcsr_idx_in_row = idx_in_row;
    meta.bcsr_global_block_index = global_block_index;
    meta.has_single_cb = (cb != nullptr);
    meta.single_cb = std::move(cb);
    meta.is_weight = true;
    meta.count_weight_read = true;
    // 诊断：blockdata 读取发起地址/跨度
    static int dbg_block_issue = 0;
    if (diag_out_ && diag_node_id_ == 0 && dbg_block_issue < 64) {
        diag_out_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-block-issue] node=%d core=%d pre=%u post=%u block_row=%u block_col=%u gbi=%u addr=0x%llx bytes=%zu aligned=(0x%llx,%zu off=%zu) off=(%u,%u) start=%u idx_in_row=%u\n",
            diag_node_id_, diag_core_id_, pre_global, post_local, block_row, block_col, global_block_index,
            (unsigned long long)addr, block_bytes,
            (unsigned long long)req_addr, req_size, slice_off,
            intra_row, intra_col, start, idx_in_row);
        ++dbg_block_issue;
    }
    (void)issueRead_(std::move(meta));
}

void WeightMemorySubsystem::maybeIssueBcsrRowptrPrefetch_() {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!mem_access_) return;
    if (orch_.bcsr_mgr->isRowptrReady()) return;
    if (orch_.bcsr_mgr->isRowptrReadPending()) return;
    if (now_cycle_ < orch_.memory_warmup_cycles) return;
    if (orch_.loader_barrier_cycles != 0 && now_cycle_ < orch_.loader_barrier_cycles) return;

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t nBlockRows = (br ? ((orch_.num_neurons + br - 1) / br) : 0);
    const size_t bytes = static_cast<size_t>(nBlockRows + 1) * sizeof(uint32_t);
    const uint64_t addr = orch_.bcsr_mgr->rowptrAddr();
    uint64_t req_addr = addr;
    size_t req_size = bytes;
    size_t slice_off = 0;
    prepareAlignedRead(addr, bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);

    PendingMeta meta{};
    meta.address = req_addr;
    meta.size = req_size;
    meta.orig_address = addr;
    meta.orig_size = bytes;
    meta.slice_offset = slice_off;
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 1;
    meta.is_weight = true;
    meta.count_weight_read = false;
    if (diag_out_) {
        diag_out_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-rowptr-issue] node=%d core=%d addr=0x%llx bytes=%zu aligned=(0x%llx,%zu off=%zu) nBlockRows=%u\n",
            diag_node_id_, diag_core_id_,
            (unsigned long long)addr, bytes,
            (unsigned long long)req_addr, req_size, slice_off,
            nBlockRows);
    }
    if (issueRead_(std::move(meta)) != 0) {
        orch_.bcsr_mgr->setRowptrReadPending(true);
    }
}

void WeightMemorySubsystem::bcsrPrefetchAll_() {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!orch_.bcsr_prefetch_all) return;
    if (bcsr_prefetch_issued_) return;
    if (!mem_access_) return;
    if (!orch_.bcsr_mgr->isRowptrReady()) return;

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t nBlockRows = (br ? ((orch_.num_neurons + br - 1) / br) : 0);
    for (uint32_t block_row = 0; block_row < nBlockRows; ++block_row) {
        uint32_t start = 0;
        uint32_t end = 0;
        if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) break;
        if (end <= start) continue;

        std::vector<uint32_t> cached_cols;
        if (orch_.bcsr_mgr->rowIndexGet(block_row, cached_cols)) {
            bcsrPrefetchRowBlocks_(block_row, cached_cols, start);
            continue;
        }

        const uint32_t block_count = end - start;
        const size_t bytes = orch_.bcsr_mgr->colIndexBytes(block_count);
        const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(start);
        uint64_t req_addr = addr;
        size_t req_size = bytes;
        size_t slice_off = 0;
        prepareAlignedRead(addr, bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);
        PendingMeta meta{};
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = bytes;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 2;
        meta.bcsr_block_row = block_row;
        meta.bcsr_row_start = start;
        meta.bcsr_prefetch_all = true;
        meta.bcsr_target_block_col = UINT32_MAX;
        meta.is_weight = true;
        meta.count_weight_read = false;
        (void)issueRead_(std::move(meta));
    }
    bcsr_prefetch_issued_ = true;
}

void WeightMemorySubsystem::bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start) {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!mem_access_) return;
    const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
    for (size_t i = 0; i < cols.size(); ++i) {
        uint32_t block_col = cols[i];
        if (orch_.bcsr_mgr->hasBlock(block_row, block_col)) continue;
        const uint32_t global_block_index = row_start + static_cast<uint32_t>(i);
        const uint64_t addr = orch_.bcsr_mgr->blockDataAddr(global_block_index);
        uint64_t req_addr = addr;
        size_t req_size = block_bytes;
        size_t slice_off = 0;
        prepareAlignedRead(addr, block_bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);
        PendingMeta meta{};
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = block_bytes;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 3;
        meta.bcsr_block_row = block_row;
        meta.bcsr_target_block_col = block_col;
        meta.bcsr_row_start = row_start;
        meta.bcsr_idx_in_row = static_cast<uint32_t>(i);
        meta.bcsr_global_block_index = global_block_index;
        meta.bcsr_prefetch_all = true;
        meta.is_weight = true;
        meta.count_weight_read = true;
        (void)issueRead_(std::move(meta));
    }
}

void WeightMemorySubsystem::bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk) {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (blk.empty()) return;
    if (orch_.weights_cols == 0) return;
    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
    const uint32_t row_base = block_row * br;
    const uint32_t col_base = block_col * bc;
    for (uint32_t rr = 0; rr < br; ++rr) {
        const uint32_t post_local = row_base + rr;
        if (post_local >= orch_.num_neurons) break;
        for (uint32_t cc = 0; cc < bc; ++cc) {
            const uint32_t pre_global = col_base + cc;
            if (pre_global >= orch_.weights_cols) break;
            const size_t idx = static_cast<size_t>(rr) * static_cast<size_t>(bc) + cc;
            if (idx >= blk.size()) break;
            const float w = blk[idx];
            const uint64_t key = static_cast<uint64_t>(post_local) * static_cast<uint64_t>(orch_.weights_cols) +
                                 static_cast<uint64_t>(pre_global);
            if (orch_.cache_put) orch_.cache_put(key, w);
            else if (cache_put_fn_) cache_put_fn_(key, w);
        }
    }
}

void WeightMemorySubsystem::handleReadResp_(uint64_t req_id, uint64_t addr, PendingMeta meta, std::vector<uint8_t>&& data) {
    (void)req_id;
    meta.address = addr;

    if (orch_.report_mem_latency && now_cycle_ >= meta.issue_cycle) {
        orch_.report_mem_latency(static_cast<uint64_t>(now_cycle_ - meta.issue_cycle), meta.is_weight);
    }
    if (meta.scheme1_prefetch && orch_.on_scheme1_prefetch_resp) {
        orch_.on_scheme1_prefetch_resp();
    }

    const std::vector<uint8_t>& bytes = data;
    const size_t float_count = bytes.size() / sizeof(float);
    const float* fptr = (float_count > 0) ? reinterpret_cast<const float*>(bytes.data()) : nullptr;

    // ---- BCSR ----
    if (meta.bcsr_kind == 1) {
        // rowptr
        if (orch_.bcsr_mgr) {
            orch_.bcsr_mgr->setRowptrReadPending(false);
            const size_t slice_off = meta.slice_offset;
            const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
            const bool slice_ok = (bytes.size() >= slice_off + slice_len);
            const uint8_t* slice_ptr = slice_ok ? (bytes.data() + slice_off) : bytes.data();
            const size_t slice_bytes = slice_ok ? slice_len : bytes.size();
            const bool ok = orch_.bcsr_mgr->installRowptrFromBytes(slice_ptr, slice_bytes, orch_.num_neurons);
            if (!ok && orch_.ensure_rowptr_ready_or_fatal) {
                orch_.ensure_rowptr_ready_or_fatal("rowptr load failed (dram)");
            }
            if (orch_.bcsr_mgr->isRowptrReady()) {
                // Rowptr 就绪后触发一次预取（可选）并尝试恢复 Apply 阶段发起
                bcsrPrefetchAll_();
                if (orch_.resume_issue_after_rowptr_ready) {
                    orch_.resume_issue_after_rowptr_ready();
                }
            }
        }
        return;
    }

    if (meta.bcsr_kind == 2) {
        // colidx
        if (!orch_.bcsr_mgr) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        if (bytes.size() < meta.size) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const size_t slice_off = meta.slice_offset;
        const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
        if (bytes.size() < slice_off + slice_len) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const uint8_t* slice = bytes.data() + slice_off;
        const uint32_t idx_bytes = orch_.bcsr_mgr->effectiveIdxBytes();
        const size_t n = idx_bytes ? (slice_len / idx_bytes) : 0;
        std::vector<uint32_t> cols(n);
        if (idx_bytes == 2) {
            for (size_t i = 0; i < n; ++i) cols[i] = reinterpret_cast<const uint16_t*>(slice)[i];
        } else {
            for (size_t i = 0; i < n; ++i) cols[i] = reinterpret_cast<const uint32_t*>(slice)[i];
        }
        // 诊断：node0/core0 仅采样部分 colidx 内容，验证与文件偏移一致
        static int dbg_colidx_dump = 0;
        if (diag_debug_ && diag_out_ && diag_node_id_ == 0 && diag_core_id_ == 0 && dbg_colidx_dump < 16 && !cols.empty()) {
            const size_t dump = std::min<size_t>(cols.size(), 8);
            std::string first_vals;
            for (size_t i = 0; i < dump; ++i) {
                if (i) first_vals.push_back(',');
                first_vals += std::to_string(cols[i]);
            }
            const size_t dump_bytes = std::min<size_t>(slice_len, 16);
            std::string first_bytes;
            for (size_t i = 0; i < dump_bytes; ++i) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "%02x", slice[i]);
                if (i) first_bytes.push_back(' ');
                first_bytes += buf;
            }
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr-colidx-dump] node=0 core=0 block_row=%u start=%u end=%u count=%zu first=%s idx_bytes=%u bytes=[%s] slice_off=%zu\n",
                meta.bcsr_block_row, meta.bcsr_row_start, meta.bcsr_row_start + (uint32_t)cols.size(),
                cols.size(), first_vals.c_str(), idx_bytes, first_bytes.c_str(), slice_off);
            ++dbg_colidx_dump;
        }
        orch_.bcsr_mgr->rowIndexPut(meta.bcsr_block_row, std::move(cols));

        std::vector<uint32_t> cached_cols;
        (void)orch_.bcsr_mgr->rowIndexGet(meta.bcsr_block_row, cached_cols);
        if (meta.bcsr_prefetch_all) {
            bcsrPrefetchRowBlocks_(meta.bcsr_block_row, cached_cols, meta.bcsr_row_start);
            return;
        }

        uint32_t idx_in_row = 0;
        bool found = false;
        for (size_t i = 0; i < cached_cols.size(); ++i) {
            if (cached_cols[i] == meta.bcsr_target_block_col) {
                idx_in_row = static_cast<uint32_t>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        std::vector<float> blk;
        if (orch_.bcsr_mgr->blockGet(meta.bcsr_block_row, meta.bcsr_target_block_col, blk)) {
            const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
            const uint32_t off = meta.bcsr_intra_row * bc + meta.bcsr_intra_col;
            float w = (off < blk.size()) ? blk[off] : 0.0f;
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(applyWeightGuards(orch_, w));
            return;
        }
        const uint32_t start = meta.bcsr_row_start;
        const uint32_t global_block_index = start + idx_in_row;
        const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
        const uint64_t block_addr = orch_.bcsr_mgr->blockDataAddr(global_block_index);
        uint64_t req_addr = block_addr;
        size_t req_size = block_bytes;
        size_t slice_off2 = 0;
        prepareAlignedRead(block_addr, block_bytes, orch_.line_size_bytes, req_addr, req_size, slice_off2);
        PendingMeta next{};
        next.address = req_addr;
        next.size = req_size;
        next.orig_address = block_addr;
        next.orig_size = block_bytes;
        next.slice_offset = slice_off2;
        next.issue_cycle = now_cycle_;
        next.bcsr_kind = 3;
        next.bcsr_block_row = meta.bcsr_block_row;
        next.bcsr_target_block_col = meta.bcsr_target_block_col;
        next.bcsr_intra_row = meta.bcsr_intra_row;
        next.bcsr_intra_col = meta.bcsr_intra_col;
        next.bcsr_row_start = start;
        next.bcsr_idx_in_row = idx_in_row;
        next.bcsr_global_block_index = global_block_index;
        next.has_single_cb = meta.has_single_cb;
        next.single_cb = std::move(meta.single_cb);
        next.is_weight = true;
        next.count_weight_read = true;
        (void)issueRead_(std::move(next));
        return;
    }

    if (meta.bcsr_kind == 3) {
        // block data
        if (!orch_.bcsr_mgr) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
        const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
        const size_t n = static_cast<size_t>(br) * static_cast<size_t>(bc);
        std::vector<float> blk(n, 0.0f);
        const size_t expect_bytes = n * sizeof(float);
        const size_t slice_off = meta.slice_offset;
        const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
        const bool ok = (bytes.size() >= slice_off + slice_len) && (slice_len >= expect_bytes);
        if (diag_out_ && diag_node_id_ == 0 && diag_core_id_ == 0) {
            static int dbg_block_resp = 0;
            if (dbg_block_resp < 2048) {
                float f0 = 0.0f;
                if (bytes.size() >= slice_off + sizeof(float)) {
                    std::memcpy(&f0, bytes.data() + slice_off, sizeof(float));
                }
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr-block-resp] node=%d core=%d block_row=%u block_col=%u gbi=%u addr=0x%llx bytes=%zu slice_off=%zu f0=%.6f ok=%d\n",
                    diag_node_id_, diag_core_id_, meta.bcsr_block_row, meta.bcsr_target_block_col,
                    meta.bcsr_global_block_index, (unsigned long long)meta.address,
                    bytes.size(), slice_off, f0, ok ? 1 : 0);
                ++dbg_block_resp;
            }
        }
        if (ok && expect_bytes > 0) {
            std::memcpy(blk.data(), bytes.data() + slice_off, expect_bytes);
            orch_.bcsr_mgr->blockPut(meta.bcsr_block_row, meta.bcsr_target_block_col, std::move(blk));
            std::vector<float> cached_blk;
            (void)orch_.bcsr_mgr->blockGet(meta.bcsr_block_row, meta.bcsr_target_block_col, cached_blk);
            bcsrPopulateWeightCache_(meta.bcsr_block_row, meta.bcsr_target_block_col, cached_blk);

            const uint32_t off = meta.bcsr_intra_row * bc + meta.bcsr_intra_col;
            float w = (off < cached_blk.size()) ? cached_blk[off] : 0.0f;
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(applyWeightGuards(orch_, w));
        } else {
            // 兜底：仅回调单个权重（不强行回填整个块），避免破坏性能/复杂性
            if (meta.has_single_cb && meta.single_cb) {
                float w = 0.0f;
                if (orch_.read_bcsr_from_file) {
                    const uint32_t post_local = meta.bcsr_block_row * br + meta.bcsr_intra_row;
                    const uint32_t pre_global = meta.bcsr_target_block_col * bc + meta.bcsr_intra_col;
                    w = orch_.read_bcsr_from_file(post_local, pre_global);
                }
                meta.single_cb(applyWeightGuards(orch_, w));
            }
        }
        return;
    }

    // ---- Dense ----
    const uint32_t width = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
    if (width > 0 && fptr) {
        for (size_t i = 0; i < float_count; ++i) {
            const uint32_t col_idx = meta.post_start + static_cast<uint32_t>(i);
            if (col_idx >= width) break;
            float w = fptr[i];
            if (orch_.readresp_zero_fallback && w == 0.0f) w = orch_.init_default_weight;
            const uint64_t key = static_cast<uint64_t>(meta.pre) * static_cast<uint64_t>(width) +
                                 static_cast<uint64_t>(col_idx);
            if (orch_.cache_put) orch_.cache_put(key, w);
            else if (cache_put_fn_) cache_put_fn_(key, w);
        }
    }
    if (meta.has_single_cb && meta.single_cb) {
        float w = 0.0f;
        const uint32_t target_col = meta.cb_post;
        if (fptr && width > 0 &&
            target_col >= meta.post_start &&
            (target_col - meta.post_start) < float_count &&
            target_col < width) {
            w = fptr[static_cast<size_t>(target_col - meta.post_start)];
        }
        if (orch_.readresp_zero_fallback && w == 0.0f) w = orch_.init_default_weight;
        meta.single_cb(w);
    }
}

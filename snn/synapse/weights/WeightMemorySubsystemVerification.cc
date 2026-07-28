// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "WeightMemorySubsystem.h"

#include "SnnBcsrWeightManager.h"
#include "SnnDLStringUtil.h"

#include <sst/core/output.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace SST;
using namespace SST::SnnDL;

namespace {

std::string resolveWeightsTemplatePath(const std::string& path_template,
                                       uint32_t pe,
                                       uint32_t core) {
    if (path_template.empty()) return "";
    return resolvePeCoreTemplate(path_template, pe, core);
}

} // namespace

bool WeightMemorySubsystem::byteExactVerifyEnabled_() const {
    if (!orch_.byte_exact_verify_enable) return false;
    return toLowerCopy(orch_.byte_exact_verify_mode) == "dense_rowcol_v1";
}

float WeightMemorySubsystem::expectedDenseWeight_(uint32_t row, uint32_t col) const {
    const uint64_t v =
        static_cast<uint64_t>(row) * static_cast<uint64_t>(orch_.byte_exact_verify_row_scale) +
        static_cast<uint64_t>(col);
    return static_cast<float>(v);
}

void WeightMemorySubsystem::verifyDenseReadBytes_(uint64_t addr, size_t req_size, const std::vector<uint8_t>& bytes) {
    if (!byteExactVerifyEnabled_()) return;
    // Dense-only; BCSR uses file-backed/structured format and has its own invariants.
    if (orch_.use_bcsr) return;
    if (orch_.num_neurons == 0) return;
    const uint32_t width = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
    if (width == 0) return;
    if (orch_.base_addr == 0) return;
    if (addr < orch_.base_addr) return;

    SST::Output* out = diagOutOrFallback_();

    if (bytes.size() != req_size) {
        byte_exact_mismatch_count_ += 1;
        if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
            byte_exact_mismatch_logged_ += 1;
            out->verbose(CALL_INFO, 0, 0,
                "[byte-exact] size-mismatch node=%u core=%u window=%u addr=0x%llx got=%zu req=%zu\n",
                orch_.node_id, orch_.core_id, window_seq_,
                (unsigned long long)addr, bytes.size(), req_size);
        }
    }

    const uint64_t off_bytes = addr - orch_.base_addr;
    if ((off_bytes & 0x3ull) != 0ull) {
        byte_exact_mismatch_count_ += 1;
        out->fatal(CALL_INFO, -1,
            "❌ [byte-exact] unaligned addr: node=%u core=%u window=%u base=0x%llx addr=0x%llx off=%" PRIu64 "\n",
            orch_.node_id, orch_.core_id, window_seq_,
            (unsigned long long)orch_.base_addr,
            (unsigned long long)addr,
            off_bytes);
    }

    const size_t nbytes = bytes.size();
    const size_t nfloat = nbytes / 4u;

    if (dense_phys_enable_) {
        const uint64_t phys_total_bytes = dense_phys_.total_bytes;
        const uint64_t row_bytes_logical = static_cast<uint64_t>(dense_phys_.row_bytes_logical);
        const uint64_t row_stride_bytes = static_cast<uint64_t>(dense_phys_.row_stride_bytes);
        const uint64_t group_stride_bytes = static_cast<uint64_t>(dense_phys_.group_stride_bytes);
        const uint64_t rows_per_dram_row = static_cast<uint64_t>(dense_phys_.rows_per_dram_row);
        const uint64_t rows_total = static_cast<uint64_t>(orch_.num_neurons);

        for (size_t i = 0; i < nfloat; ++i) {
            const uint64_t p = off_bytes + static_cast<uint64_t>(i) * 4ull;
            if (p >= phys_total_bytes) {
                byte_exact_mismatch_count_ += 1;
                if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
                    byte_exact_mismatch_logged_ += 1;
                    out->verbose(CALL_INFO, 0, 0,
                        "[byte-exact] oob-phys node=%u core=%u window=%u addr=0x%llx float_i=%zu off=%" PRIu64 " phys_total=%" PRIu64 "\n",
                        orch_.node_id, orch_.core_id, window_seq_,
                        (unsigned long long)addr, i, p, phys_total_bytes);
                }
                continue;
            }

            const uint64_t group = (group_stride_bytes != 0) ? (p / group_stride_bytes) : 0;
            const uint64_t within_group = (group_stride_bytes != 0) ? (p % group_stride_bytes) : p;
            const uint64_t row_in_group = (row_stride_bytes != 0) ? (within_group / row_stride_bytes) : 0;
            const uint64_t within_row = (row_stride_bytes != 0) ? (within_group % row_stride_bytes) : within_group;
            const uint64_t row64 = group * rows_per_dram_row + row_in_group;

            uint8_t expect_b[4] = {0, 0, 0, 0};
            uint32_t row = 0;
            uint32_t col = 0;
            bool padding = true;
            if (row64 < rows_total && within_row < row_bytes_logical) {
                padding = false;
                row = static_cast<uint32_t>(row64);
                col = static_cast<uint32_t>(within_row / 4ull);
                const float expect_f = expectedDenseWeight_(row, col);
                std::memcpy(expect_b, &expect_f, sizeof(expect_b));
            }

            const uint8_t* got_b = bytes.data() + i * 4u;
            if (std::memcmp(got_b, expect_b, 4u) != 0) {
                byte_exact_mismatch_count_ += 1;
                if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
                    byte_exact_mismatch_logged_ += 1;
                    if (padding) {
                        out->verbose(CALL_INFO, 0, 0,
                            "[byte-exact] pad-mismatch node=%u core=%u window=%u addr=0x%llx float_i=%zu off=%" PRIu64 " row=%" PRIu64 " within_row=%" PRIu64 " got=[%02x %02x %02x %02x] expect=[%02x %02x %02x %02x]\n",
                            orch_.node_id, orch_.core_id, window_seq_,
                            (unsigned long long)addr, i, p, row64, within_row,
                            got_b[0], got_b[1], got_b[2], got_b[3],
                            expect_b[0], expect_b[1], expect_b[2], expect_b[3]);
                    } else {
                        float got_f = 0.0f;
                        float expect_f = 0.0f;
                        std::memcpy(&got_f, got_b, sizeof(got_f));
                        std::memcpy(&expect_f, expect_b, sizeof(expect_f));
                        out->verbose(CALL_INFO, 0, 0,
                            "[byte-exact] mismatch node=%u core=%u window=%u addr=0x%llx float_i=%zu row=%u col=%u got_f=%.9g expect_f=%.9g got=[%02x %02x %02x %02x] expect=[%02x %02x %02x %02x]\n",
                            orch_.node_id, orch_.core_id, window_seq_,
                            (unsigned long long)addr,
                            i, row, col,
                            got_f, expect_f,
                            got_b[0], got_b[1], got_b[2], got_b[3],
                            expect_b[0], expect_b[1], expect_b[2], expect_b[3]);
                    }
                }
                if (byte_exact_mismatch_count_ >= orch_.byte_exact_verify_max_mismatch) {
                    out->fatal(CALL_INFO, -1,
                        "❌ [byte-exact] too many mismatches: node=%u core=%u window=%u mismatches=%u max=%u\n",
                        orch_.node_id, orch_.core_id, window_seq_,
                        byte_exact_mismatch_count_, orch_.byte_exact_verify_max_mismatch);
                }
            }
        }

        if (nfloat > 0) byte_exact_verified_reads_ += 1;
        return;
    }

    const uint64_t total_floats = static_cast<uint64_t>(orch_.num_neurons) * static_cast<uint64_t>(width);
    const uint64_t start_float = off_bytes / 4ull;
    for (size_t i = 0; i < nfloat; ++i) {
        const uint64_t idx = start_float + static_cast<uint64_t>(i);
        if (idx >= total_floats) {
            byte_exact_mismatch_count_ += 1;
            if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
                byte_exact_mismatch_logged_ += 1;
                out->verbose(CALL_INFO, 0, 0,
                    "[byte-exact] oob idx node=%u core=%u window=%u addr=0x%llx float_i=%zu idx=%" PRIu64 " total=%" PRIu64 "\n",
                    orch_.node_id, orch_.core_id, window_seq_,
                    (unsigned long long)addr, i, idx, total_floats);
            }
            continue;
        }
        const uint32_t row = static_cast<uint32_t>(idx / static_cast<uint64_t>(width));
        const uint32_t col = static_cast<uint32_t>(idx % static_cast<uint64_t>(width));
        const float expect_f = expectedDenseWeight_(row, col);
        uint8_t expect_b[4];
        std::memcpy(expect_b, &expect_f, sizeof(expect_b));
        const uint8_t* got_b = bytes.data() + i * 4u;
        if (std::memcmp(got_b, expect_b, 4u) != 0) {
            byte_exact_mismatch_count_ += 1;
            if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
                byte_exact_mismatch_logged_ += 1;
                float got_f = 0.0f;
                std::memcpy(&got_f, got_b, sizeof(got_f));
                out->verbose(CALL_INFO, 0, 0,
                    "[byte-exact] mismatch node=%u core=%u window=%u addr=0x%llx float_i=%zu row=%u col=%u got_f=%.9g expect_f=%.9g got=[%02x %02x %02x %02x] expect=[%02x %02x %02x %02x]\n",
                    orch_.node_id, orch_.core_id, window_seq_,
                    (unsigned long long)addr,
                    i, row, col,
                    got_f, expect_f,
                    got_b[0], got_b[1], got_b[2], got_b[3],
                    expect_b[0], expect_b[1], expect_b[2], expect_b[3]);
            }
            if (byte_exact_mismatch_count_ >= orch_.byte_exact_verify_max_mismatch) {
                out->fatal(CALL_INFO, -1,
                    "❌ [byte-exact] too many mismatches: node=%u core=%u window=%u mismatches=%u max=%u\n",
                    orch_.node_id, orch_.core_id, window_seq_,
                    byte_exact_mismatch_count_, orch_.byte_exact_verify_max_mismatch);
            }
        }
    }

    // Count a response as "verified" only when we could meaningfully interpret it as dense floats.
    if (nfloat > 0) byte_exact_verified_reads_ += 1;
}

void WeightMemorySubsystem::verifyDenseEdgeWeight_(uint32_t pre_global, uint32_t post_local, uint32_t count, float weight) {
    if (!byteExactVerifyEnabled_()) return;
    if (orch_.use_bcsr) return;
    if (!orch_.accessor) return;

    SST::Output* out = diagOutOrFallback_();

    uint32_t req_pre = 0;
    uint32_t req_post = 0;
    uint64_t cache_key = 0;
    if (!orch_.accessor->resolve(pre_global, post_local, req_pre, req_post, cache_key)) return;

    const uint32_t row = orch_.use_post_row_pre_col ? req_post : req_pre;
    const uint32_t col = orch_.use_post_row_pre_col ? req_pre : req_post;
    const float expect = expectedDenseWeight_(row, col);

    if (!(weight == expect)) {
        byte_exact_mismatch_count_ += 1;
        if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
            byte_exact_mismatch_logged_ += 1;
            out->verbose(CALL_INFO, 0, 0,
                "[byte-exact] edge-mismatch node=%u core=%u window=%u pre=%u post=%u req_pre=%u req_post=%u row=%u col=%u got=%.9g expect=%.9g count=%u\n",
                orch_.node_id, orch_.core_id, window_seq_,
                pre_global, post_local, req_pre, req_post, row, col,
                weight, expect, count);
        }
        if (byte_exact_mismatch_count_ >= orch_.byte_exact_verify_max_mismatch) {
            out->fatal(CALL_INFO, -1,
                "❌ [byte-exact] too many mismatches: node=%u core=%u window=%u mismatches=%u max=%u\n",
                orch_.node_id, orch_.core_id, window_seq_,
                byte_exact_mismatch_count_, orch_.byte_exact_verify_max_mismatch);
        }
    }

    // dv correctness (should be exact under dense_rowcol_v1 range constraints).
    const float dv = weight * static_cast<float>(count);
    const float dv_expect = expect * static_cast<float>(count);
    if (!(dv == dv_expect)) {
        byte_exact_mismatch_count_ += 1;
        if (byte_exact_mismatch_logged_ < orch_.byte_exact_verify_max_mismatch) {
            byte_exact_mismatch_logged_ += 1;
            out->verbose(CALL_INFO, 0, 0,
                "[byte-exact] dv-mismatch node=%u core=%u window=%u pre=%u post=%u dv=%.9g dv_expect=%.9g count=%u\n",
                orch_.node_id, orch_.core_id, window_seq_,
                pre_global, post_local, dv, dv_expect, count);
        }
        if (byte_exact_mismatch_count_ >= orch_.byte_exact_verify_max_mismatch) {
            out->fatal(CALL_INFO, -1,
                "❌ [byte-exact] too many mismatches: node=%u core=%u window=%u mismatches=%u max=%u\n",
                orch_.node_id, orch_.core_id, window_seq_,
                byte_exact_mismatch_count_, orch_.byte_exact_verify_max_mismatch);
        }
    }

    byte_exact_verified_edges_ += 1;
}

void WeightMemorySubsystem::emitByteExactPassMarker_(const char* where, uint32_t seq) {
    if (!byteExactVerifyEnabled_()) return;
    if (byte_exact_pass_logged_) return;

    SST::Output* out = diagOutOrFallback_();

    if (byte_exact_mismatch_count_ != 0) {
        out->fatal(CALL_INFO, -1,
            "❌ BYTE_EXACT_VERIFY: mismatches=%u (expected 0) node=%u core=%u where=%s seq=%u\n",
            byte_exact_mismatch_count_,
            orch_.node_id, orch_.core_id,
            (where ? where : "?"), seq);
    }
    if (byte_exact_verified_reads_ == 0 || byte_exact_verified_edges_ == 0) {
        out->fatal(CALL_INFO, -1,
            "❌ BYTE_EXACT_VERIFY: no effective verification (reads=%" PRIu64 " edges=%" PRIu64 ") node=%u core=%u where=%s seq=%u\n",
            byte_exact_verified_reads_, byte_exact_verified_edges_,
            orch_.node_id, orch_.core_id,
            (where ? where : "?"), seq);
    }
    out->verbose(CALL_INFO, 0, 0,
        "BYTE_EXACT_VERIFY: PASS mode=%s node=%u core=%u where=%s seq=%u verified_reads=%" PRIu64 " verified_edges=%" PRIu64 "\n",
        orch_.byte_exact_verify_mode.c_str(),
        orch_.node_id, orch_.core_id,
        (where ? where : "?"), seq,
        byte_exact_verified_reads_, byte_exact_verified_edges_);
    byte_exact_pass_logged_ = true;
}

bool WeightMemorySubsystem::bcsrSemanticVerifyEnabled_() const {
    if (!orch_.bcsr_semantic_verify_enable) return false;
    if (!orch_.use_bcsr) return false;
    return true;
}

void WeightMemorySubsystem::verifyBcsrEdgeWeight_(uint32_t pre_global, uint32_t post_local, float weight) {
    if (!bcsrSemanticVerifyEnabled_()) return;
    if (bcsr_sem_pass_logged_) return;
    if (orch_.bcsr_semantic_verify_max_edges == 0) return;
    if (bcsr_sem_verified_edges_ >= static_cast<uint64_t>(orch_.bcsr_semantic_verify_max_edges)) return;

    SST::Output* out = diagOutOrFallback_();

    bool have_ref = false;
    float ref_raw = 0.0f;
    if (orch_.read_bcsr_from_file) {
        ref_raw = orch_.read_bcsr_from_file(post_local, pre_global);
        have_ref = true;
    } else if (orch_.bcsr_mgr && !orch_.weights_template.empty() && orch_.base_addr != 0) {
        // Fallback: direct file read (independent of platform/core/workload).
        const std::string path = resolveWeightsTemplatePath(orch_.weights_template, orch_.node_id, orch_.core_id);
        if (!path.empty()) {
            std::ifstream fin(path, std::ios::in | std::ios::binary);
            if (fin.good()) {
                const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
                const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
                const uint32_t idxB = orch_.bcsr_mgr->effectiveIdxBytes();
                const uint32_t valB = orch_.bcsr_mgr->effectiveValBytes();
                if (br != 0 && bc != 0 && (idxB == 2 || idxB == 4) && valB == 4) {
                    const uint32_t block_row = post_local / br;
                    const uint32_t intra_row = post_local % br;
                    const uint32_t blk_col = pre_global / bc;
                    const uint32_t intra_col = pre_global % bc;

                    const uint64_t rp_off = orch_.bcsr_mgr->rowptrAddr() - orch_.base_addr;
                    const uint64_t ci_off = orch_.bcsr_mgr->colidxAddr() - orch_.base_addr;
                    const uint64_t bd_off = orch_.bcsr_mgr->blockdataAddr() - orch_.base_addr;

                    uint32_t start = 0;
                    uint32_t end = 0;
                    fin.seekg(static_cast<std::streamoff>(rp_off + static_cast<uint64_t>(block_row) * sizeof(uint32_t)), std::ios::beg);
                    fin.read(reinterpret_cast<char*>(&start), 4);
                    fin.read(reinterpret_cast<char*>(&end), 4);
                    if (fin.good() && end > start) {
                        int idx_in_row = -1;
                        for (uint32_t j = 0; j < (end - start); ++j) {
                            fin.seekg(static_cast<std::streamoff>(ci_off + static_cast<uint64_t>(start + j) * idxB), std::ios::beg);
                            uint32_t colv = 0;
                            if (idxB == 2) {
                                uint16_t v = 0;
                                fin.read(reinterpret_cast<char*>(&v), 2);
                                colv = v;
                            } else {
                                fin.read(reinterpret_cast<char*>(&colv), 4);
                            }
                            if (!fin.good()) break;
                            if (colv == blk_col) { idx_in_row = static_cast<int>(j); break; }
                        }
                        if (fin.good() && idx_in_row >= 0) {
                            const size_t blk_bytes = static_cast<size_t>(br) * static_cast<size_t>(bc) * sizeof(float);
                            fin.seekg(static_cast<std::streamoff>(bd_off + static_cast<uint64_t>(start + static_cast<uint32_t>(idx_in_row)) * blk_bytes), std::ios::beg);
                            std::vector<float> blk(static_cast<size_t>(br) * static_cast<size_t>(bc), 0.0f);
                            fin.read(reinterpret_cast<char*>(blk.data()), static_cast<std::streamsize>(blk_bytes));
                            if (fin.good()) {
                                const uint32_t off = intra_row * bc + intra_col;
                                ref_raw = (off < blk.size()) ? blk[off] : 0.0f;
                                have_ref = true;
                            }
                        } else {
                            // Block not present -> weight is logically 0.
                            ref_raw = 0.0f;
                            have_ref = true;
                        }
                    }
                } else {
                    bcsr_sem_inconclusive_ = true;
                    if (bcsr_sem_inconclusive_reason_.empty()) bcsr_sem_inconclusive_reason_ = "unsupported_bcsr_val_or_idx_bytes";
                    return;
                }
            }
        }
    }

    if (!have_ref) {
        bcsr_sem_inconclusive_ = true;
        if (bcsr_sem_inconclusive_reason_.empty()) bcsr_sem_inconclusive_reason_ = "missing_ref_reader";
        return;
    }
    const float ref = applyWeightGuards_(ref_raw);

    const float abs_tol = orch_.bcsr_semantic_verify_abs_tol;
    const float rel_tol = orch_.bcsr_semantic_verify_rel_tol;
    const float diff = std::fabs(weight - ref);
    const float tol = abs_tol + rel_tol * std::fabs(ref);
    const bool ok = std::isfinite(weight) && std::isfinite(ref) && (diff <= tol);
    if (!ok) {
        bcsr_sem_mismatch_count_ += 1;
        if (bcsr_sem_mismatch_logged_ < orch_.bcsr_semantic_verify_max_mismatch) {
            bcsr_sem_mismatch_logged_ += 1;
            out->verbose(CALL_INFO, 0, 0,
                "BCSR_SEMANTIC_VERIFY: MISMATCH node=%u core=%u window=%u pre=%u post=%u got=%.9g ref=%.9g diff=%.9g tol=%.9g\n",
                orch_.node_id, orch_.core_id, window_seq_,
                pre_global, post_local,
                weight, ref, diff, tol);
        }
        if (bcsr_sem_mismatch_count_ >= orch_.bcsr_semantic_verify_max_mismatch) {
            out->fatal(CALL_INFO, -1,
                "❌ BCSR_SEMANTIC_VERIFY: too many mismatches node=%u core=%u mismatches=%u max=%u\n",
                orch_.node_id, orch_.core_id,
                bcsr_sem_mismatch_count_, orch_.bcsr_semantic_verify_max_mismatch);
        }
        return;
    }

    bcsr_sem_verified_edges_ += 1;
}

void WeightMemorySubsystem::emitBcsrSemanticVerifyMarker_(const char* where, uint32_t seq) {
    if (!bcsrSemanticVerifyEnabled_()) return;
    if (bcsr_sem_pass_logged_) return;

    SST::Output* out = diagOutOrFallback_();

    if (bcsr_sem_mismatch_count_ != 0) {
        out->fatal(CALL_INFO, -1,
            "❌ BCSR_SEMANTIC_VERIFY: mismatches=%u (expected 0) node=%u core=%u where=%s seq=%u\n",
            bcsr_sem_mismatch_count_,
            orch_.node_id, orch_.core_id,
            (where ? where : "?"), seq);
    }

    if (bcsr_sem_verified_edges_ == 0) {
        const char* reason = bcsr_sem_inconclusive_reason_.empty()
                                 ? (bcsr_sem_inconclusive_ ? "inconclusive" : "no_verified_edges")
                                 : bcsr_sem_inconclusive_reason_.c_str();
        out->verbose(CALL_INFO, 0, 0,
            "BCSR_SEMANTIC_VERIFY: WARN INCONCLUSIVE node=%u core=%u where=%s seq=%u verified_edges=%" PRIu64 " reason=%s\n",
            orch_.node_id, orch_.core_id,
            (where ? where : "?"), seq,
            bcsr_sem_verified_edges_,
            reason);
        bcsr_sem_pass_logged_ = true;
        return;
    }

    out->verbose(CALL_INFO, 0, 0,
        "BCSR_SEMANTIC_VERIFY: PASS node=%u core=%u where=%s seq=%u verified_edges=%" PRIu64 " max_edges=%u\n",
        orch_.node_id, orch_.core_id,
        (where ? where : "?"), seq,
        bcsr_sem_verified_edges_,
        orch_.bcsr_semantic_verify_max_edges);
    bcsr_sem_pass_logged_ = true;
}


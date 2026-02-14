// -*- c++ -*-
// DramAwareTuner.h: per-window tuning for GAS Apply (k/budget guards)

#pragma once

#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>

#include "components/gather/apply/DramAwareModel.h"

namespace SST { namespace SnnDL { namespace gather { namespace apply {

enum class DramAwareKPolicy : uint8_t {
    Fixed = 0,
    CostBudgeted = 1,
    DensityBudgeted = 2,
};

struct DramAwareEffectivePolicy {
    bool gap_merge_enable = false;
    uint64_t gap_k_bytes = 0;
    uint64_t overfetch_budget_bytes = 0; // 0 = unlimited
};

class DramAwareTuner {
public:
    DramAwareTuner(const DramAwareParams& params, DramAwareKPolicy k_policy, uint64_t k_cap_bytes)
        : params_(params), k_policy_(k_policy), k_cap_bytes_(k_cap_bytes), model_(params) {}

    DramAwareEffectivePolicy derive(const WindowAccessSummary& summary) const {
        DramAwareEffectivePolicy out{};
        out.overfetch_budget_bytes = params_.overfetch_budget_bytes;

        const bool budget_enabled = (params_.overfetch_budget_bytes > 0);
        const bool meaningful_access = (summary.unique_line_count > 0 || summary.payload_bytes > 0);

        // Default: keep disabled unless we have either explicit cap or budget.
        out.gap_merge_enable = meaningful_access && (k_cap_bytes_ > 0 || budget_enabled);

        if (!out.gap_merge_enable) {
            out.gap_k_bytes = 0;
            return out;
        }

        if (k_policy_ == DramAwareKPolicy::Fixed) {
            out.gap_k_bytes = k_cap_bytes_;
            return out;
        }

        if (k_policy_ == DramAwareKPolicy::DensityBudgeted) {
            const double d = model_.estimateDensity(summary);
            const uint64_t base = std::max<uint32_t>(model_.lineBytes(), model_.readBurstBytes());
            const uint64_t k = static_cast<uint64_t>(base * (1.0 + 15.0 * d)); // [1..16]*base
            out.gap_k_bytes = std::min<uint64_t>(k_cap_bytes_ ? k_cap_bytes_ : k, k);
            return out;
        }

        // CostBudgeted (default): include row-miss penalty weight.
        out.gap_k_bytes = model_.recommendGapKBytes(k_cap_bytes_, summary);
        return out;
    }

    static DramAwareKPolicy parseKPolicy(const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (v == "fixed") return DramAwareKPolicy::Fixed;
        if (v == "density_budgeted") return DramAwareKPolicy::DensityBudgeted;
        return DramAwareKPolicy::CostBudgeted;
    }

private:
    DramAwareParams params_{};
    DramAwareKPolicy k_policy_ = DramAwareKPolicy::CostBudgeted;
    uint64_t k_cap_bytes_ = 0;
    DramAwareModel model_;
};

}}}} // namespace SST::SnnDL::gather::apply

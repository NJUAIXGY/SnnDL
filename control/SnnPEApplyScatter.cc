#include "SnnPESubComponent.h"
#include <algorithm>
#include <cstdint>

using namespace SST::SnnDL;

uint64_t SnnPESubComponent::applyAccumulatedWindowAndScatter_() {
    uint64_t spikes_emitted = 0;
    auto addScatterStat = [&](uint64_t delta){
        if (delta == 0) return;
        if (stat_gas_scatter_spikes_emitted_total_) stat_gas_scatter_spikes_emitted_total_->addData(delta);
    };

    std::vector<std::pair<uint32_t, float>> pairs;
    if (acc_ops_) {
        acc_ops_->collectSortedPairs(pairs);
    }

    if (!compute_core_) {
        // 理论上不会发生（构造期已配置 DefaultSnnComputeCore），但保持健壮性。
        if (acc_ops_) acc_ops_->reset();
        spikes_emitted_window_ = 0;
        if (acc_ops_ && acc_ops_->denseEnabled()) verifyDenseAccumulator_(curr_stage_seq_);
        return 0;
    }

    for (const auto& pr : pairs) {
        uint32_t post = pr.first;
        float dv = pr.second;
        if (dv == 0.0f) continue;
        applySynapticDelta_(post, dv);
    }
    // 由 compute core 在统一收敛点处理动力学+发放
    compute_core_->endCycle(static_cast<uint64_t>(total_cycles_));
    std::vector<FireEvent> fired;
    compute_core_->drainOutputs(fired, true);
    spikes_emitted += routeAndSendOutputs_(fired);

    if (acc_ops_) acc_ops_->reset();
    spikes_emitted_window_ = spikes_emitted;
    if (acc_ops_ && acc_ops_->denseEnabled()) verifyDenseAccumulator_(curr_stage_seq_);
    addScatterStat(spikes_emitted);
    return spikes_emitted;
}

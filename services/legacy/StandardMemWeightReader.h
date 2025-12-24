// -*- c++ -*-
//
// StandardMemWeightReader: StandardMem-based weight reader/controller extracted from
// SnnPESubComponent. Implements IWeightReader for compute core and centralizes
// dense/BCSR read/response handling.
//
// 说明：该模块为历史遗留/参考实现（当前默认不纳入构建）。
// 现行数据路径已由 `services/StandardMemAccess`（纯内存） + `services/WeightMemorySubsystem`（权重语义）闭环，
// 避免 services→control 的反向依赖。
//

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "SnnWeightReader.h" // IWeightReader

// 若需处理 StandardMem 回包（legacy API），需要完整类型定义
#include <sst/core/interfaces/stdMem.h>

namespace SST { namespace SnnDL {

class StandardMemWeightReader final : public IWeightReader {
public:
    StandardMemWeightReader() = default;

    // 绑定底层实现（建议使用 WeightMemorySubsystem；该 shim 自身不再窥探控制层私有成员）
    void bind(IWeightReader* impl) { impl_ = impl; }

    // IWeightReader
    void requestDense(uint32_t pre, uint32_t post, std::function<void(float)> cb) override;
    void requestBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) override;
    bool tryCache(uint64_t key, float& out) override;
    void putCache(uint64_t key, float value) override;

    // Legacy API：写回 helper（现行路径已在 control/SnnPESubComponent_mem.cc 内实现）
    bool applyLocalWeightUpdates(const std::unordered_map<uint64_t, float>& grads,
                                 float learning_rate,
                                 float weight_decay);

    // Legacy API：scheme1 预取 helper（现行路径已由 WeightMemorySubsystem::issueDensePrefetchRaw 支持）
    void scheme1PrefetchSlice(uint32_t slice_idx);

private:
    IWeightReader* impl_ = nullptr; // non-owning
};

}} // namespace SST::SnnDL

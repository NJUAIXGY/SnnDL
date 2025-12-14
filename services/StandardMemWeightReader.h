// -*- c++ -*-
//
// StandardMemWeightReader: StandardMem-based weight reader/controller extracted from
// SnnPESubComponent. Implements IWeightReader for compute core and centralizes
// dense/BCSR read/response handling. Behavior preserved; structural decoupling only.
//

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "SnnWeightReader.h" // IWeightReader

namespace SST { namespace Interfaces { class StandardMem; } }

namespace SST { namespace SnnDL {

class SnnPESubComponent; // fwd

class StandardMemWeightReader final : public IWeightReader {
public:
    StandardMemWeightReader() = default;

    void init(SnnPESubComponent* core) { core_ = core; }

    // IWeightReader
    void requestDense(uint32_t pre, uint32_t post, std::function<void(float)> cb) override;
    void requestBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) override;
    bool tryCache(uint64_t key, float& out) override;
    void putCache(uint64_t key, float value) override;

    // Handle non-Custom StandardMem responses (ReadResp/WriteResp).
    // Returns true if handled and request deleted.
    bool handleMemoryResponse(SST::Interfaces::StandardMem::Request* req);

    // Writeback helper used by compute core callback.
    bool applyLocalWeightUpdates(const std::unordered_map<uint64_t, float>& grads,
                                 float learning_rate,
                                 float weight_decay);

    // Scheme1 prefetch helper.
    void scheme1PrefetchSlice(uint32_t slice_idx);

private:
    bool prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                           uint64_t& req_addr, size_t& req_size,
                           bool& is_row, uint32_t& col_start, uint32_t& count_floats) const;
    void issueReadCommon_(uint64_t req_addr, size_t req_size,
                          bool is_row, uint32_t row, uint32_t col_start, uint32_t count_floats,
                          std::function<void(float)> single_cb, uint32_t single_col);

    SnnPESubComponent* core_ = nullptr;
};

}} // namespace SST::SnnDL


// -*- c++ -*-
//
// ReadOrchestrator: window-read issue logic extracted from SnnPESubComponent.
//
// 说明：该文件属于历史遗留/参考实现（当前默认不纳入构建）。
// 真实数据路径已由 `services/WeightMemorySubsystem` 闭环承载，避免 services→control 的反向依赖。

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sst/core/output.h>

namespace SST { namespace SnnDL {

class WeightMemorySubsystem;

struct DiagSink {
    SST::Output* output = nullptr;
    bool window_read_debug = false;

    void init(SST::Output* out, bool debug) { output = out; window_read_debug = debug; }
    bool enabled() const { return window_read_debug && output; }
    SST::Output* out() const { return output; }
    template <typename... Args>
    void log(int, const char* fmt, Args&&... args) const {
        if (!enabled()) return;
        if (auto* o = out()) {
            if constexpr (sizeof...(Args) == 0) {
                o->verbose(CALL_INFO, 0, 0, "%s", fmt);
            } else {
                o->verbose(CALL_INFO, 0, 0, fmt, std::forward<Args>(args)...);
            }
        }
    }
};

struct ReadOrchestrator {
    WeightMemorySubsystem* mem = nullptr;
    DiagSink diag_;
    void init(WeightMemorySubsystem* owner, SST::Output* out, bool window_read_debug) {
        mem = owner;
        diag_.init(out, window_read_debug);
    }
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

}} // namespace SST::SnnDL

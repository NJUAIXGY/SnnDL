// -*- c++ -*-
//
// ReadOrchestrator: window-read issue logic extracted from SnnPESubComponent.
// Behavior preserved; only structural decoupling.

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sst/core/output.h>

namespace SST { namespace SnnDL {

class SnnPESubComponent;

struct DiagSink {
    SnnPESubComponent* core = nullptr;
    void init(SnnPESubComponent* owner) { core = owner; }
    bool enabled() const;
    SST::Output* out() const;
    template <typename... Args>
    void log(int, const char* fmt, Args&&... args) const {
        if (!enabled()) return;
        if (auto* o = out()) {
            o->verbose(CALL_INFO, 0, 0, fmt, std::forward<Args>(args)...);
        }
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

}} // namespace SST::SnnDL

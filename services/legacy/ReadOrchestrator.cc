// -*- c++ -*-
//
// ReadOrchestrator: window-read issue logic extracted from SnnPESubComponent.

#include "ReadOrchestrator.h"
#include "weights/WeightMemorySubsystem.h"

using namespace SST::SnnDL;

void ReadOrchestrator::issueFromEdges() {
    if (!mem) return;
    mem->issueFromEdges();
}

void ReadOrchestrator::issueFromSets(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!mem || !posts_to_use || !pres_to_use) return;
    mem->issueFromSets(posts_to_use, pres_to_use);
}

void ReadOrchestrator::issueFallbackReadsIfNeeded(bool strict_gas_active) {
    if (!mem) return;
    mem->issueFallbackReadsIfNeeded(strict_gas_active);
}

void ReadOrchestrator::issueFromSetsBcsr(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!mem || !posts_to_use || !pres_to_use) return;
    mem->issueFromSetsBcsr(posts_to_use, pres_to_use);
}

void ReadOrchestrator::logWindowReadSummary_(
    uint32_t posts_prev, uint32_t pres_prev,
    uint32_t posts_curr, uint32_t pres_curr,
    bool fallback) const {
    diag_.log(1,
        "[diag-window-read] BeginApply: prev(posts=%u pre=%u) curr(posts=%u pre=%u) budget=%u max_out=%u fallback=%d\n",
        posts_prev, pres_prev, posts_curr, pres_curr,
        mem ? mem->budget() : 0u,
        mem ? mem->maxOutstanding() : 0u,
        fallback ? 1 : 0);
}

void ReadOrchestrator::logFallbackSwitch_() const {
    diag_.log(1,
        "[diag-window-read] BeginApply: FALLBACK to current window (prev empty, curr has data)\n");
}

void ReadOrchestrator::logIssuedStats_(uint32_t issued) const {
    diag_.log(1,
        "[diag-window-read] BeginApply: issued=%u (this window) outstanding_reqs=%u pending=%zu\n",
        issued,
        mem ? mem->outstanding() : 0u,
        mem ? mem->pendingSize() : 0u);
}

void ReadOrchestrator::logEdgeFetchStart(
    size_t prev_edges, uint32_t issued, uint32_t outstanding, uint32_t budget) const {
    diag_.log(1,
        "[diag-edge-fetch] prev_edges=%zu issued=%u outstanding=%u budget=%u\n",
        prev_edges,
        issued, outstanding, budget);
}

bool ReadOrchestrator::canIssueMoreReads_() const {
    return mem ? mem->canIssue() : false;
}

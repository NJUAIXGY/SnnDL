// -*- c++ -*-
//
// SnnWorkloadStatsModule: PE-level aggregation for snn workload receive-path counters.
//

#pragma once

#include "KeyedCounterStatsModule.h"

namespace SST { namespace SnnDL {

class SnnWorkloadStatsModule final : public KeyedCounterStatsModule {
public:
    explicit SnnWorkloadStatsModule(bool active_workload) : KeyedCounterStatsModule(active_workload) {
        addCounter("snn_tx_spike_packets_total");
        addCounter("snn_tx_spikekey_packets_total");
        addCounter("snn_tx_spiketilekey_packets_total");
        addCounter("snn_rx_spike_packets_total");
        addCounter("snn_rx_spikekey_total");
        addCounter("snn_rx_spiketilekey_total");
        addCounter("snn_rx_fastpath_packets_total");
        addCounter("snn_rx_fallback_packets_total");
        addCounter("snn_rx_decode_fail_total");
        addCounter("snn_rx_fastpath_posts_total");
        addCounter("snn_rx_fastpath_accept_total");
        addCounter("snn_rx_fastpath_reject_total");
        addCounter("snn_rx_fastpath_edges_recorded_total");
    }

    const char* name() const override { return "snn"; }
};

}} // namespace SST::SnnDL

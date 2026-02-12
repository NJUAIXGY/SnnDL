// -*- c++ -*-
//
// TensorWorkloadStatsModule: PE-level aggregation for tensor workload counters.
//

#pragma once

#include "KeyedCounterStatsModule.h"

namespace SST { namespace SnnDL {

class TensorWorkloadStatsModule final : public KeyedCounterStatsModule {
public:
    explicit TensorWorkloadStatsModule(bool active_workload) : KeyedCounterStatsModule(active_workload) {
        addCounter("tensor_mem_reads_issued_total");
        addCounter("tensor_mem_writes_issued_total");
        addCounter("tensor_mem_bytes_read_total");
        addCounter("tensor_mem_bytes_write_total");
        addCounter("tensor_compute_cycles_total");
        addCounter("tensor_mac_ops_total");
        addCounter("tensor_dma_stall_cycles_total");
        addCounter("tensor_iter_cycles_total");
        addCounter("tensor_stall_dma_budget_cycles_total");
        addCounter("tensor_stall_mem_outstanding_cycles_total");
        addCounter("tensor_stall_wait_read_cycles_total");
        addCounter("tensor_stall_wait_write_cycles_total");
        addCounter("tensor_stall_collective_cycles_total");
        addCounter("tensor_dma_cycles_total");
        addCounter("tensor_dram_bytes_total");
        addCounter("tensor_onchip_bytes_total");
        addCounter("tensor_tile_count_total");
        addCounter("tensor_collective_bytes_sent_total");
        addCounter("tensor_collective_bytes_recv_total");
        addCounter("tensor_collective_pkts_sent_total");
        addCounter("tensor_collective_pkts_recv_total");
        addCounter("tensor_collective_cycles_total");
        addCounter("tensor_pkt_sent_total");
        addCounter("tensor_pkt_recv_total");
        addCounter("tensor_pkt_bytes_sent_total");
        addCounter("tensor_pkt_bytes_recv_total");
    }

    const char* name() const override { return "tensor"; }
};

}} // namespace SST::SnnDL

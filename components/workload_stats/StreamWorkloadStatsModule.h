// -*- c++ -*-
//
// StreamWorkloadStatsModule: PE-level aggregation for stream workload counters.
//

#pragma once

#include "KeyedCounterStatsModule.h"

namespace SST { namespace SnnDL {

class StreamWorkloadStatsModule final : public KeyedCounterStatsModule {
public:
    explicit StreamWorkloadStatsModule(bool active_workload) : KeyedCounterStatsModule(active_workload) {
        addCounter("stream_mem_writes_issued_total");
        addCounter("stream_mem_reads_issued_total");
        addCounter("stream_mem_bytes_written_total");
        addCounter("stream_mem_bytes_read_total");
        addCounter("stream_mem_verify_pass_total");
        addCounter("stream_mem_verify_fail_total");
        addCounter("stream_pkt_sent_total");
        addCounter("stream_pkt_recv_total");
        addCounter("stream_pkt_bad_crc_total");
        addCounter("stream_pkt_bad_magic_total");
    }

    const char* name() const override { return "stream"; }
};

}} // namespace SST::SnnDL


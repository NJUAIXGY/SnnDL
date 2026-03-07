// -*- c++ -*-
//
// TassLfP0ReportEvent:
// - MultiCorePE -> MultiCorePE 的控制面事件：将单 core/window 的 TASS-LF P0 报告送到 block-origin reporter PE。
// - 目的：桥接 process-local registry 在 MPI 多 rank 下不可见的问题。
//

#ifndef SNNDL_TASS_LF_P0_REPORT_EVENT_H
#define SNNDL_TASS_LF_P0_REPORT_EVENT_H

#include <cstdint>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

#include "IPeAggregation.h"

namespace SST { namespace SnnDL {

class TassLfP0ReportEvent final : public SST::Event {
public:
    TassLfP0WindowReport report{};

    TassLfP0ReportEvent() = default;
    explicit TassLfP0ReportEvent(const TassLfP0WindowReport& r) : SST::Event(), report(r) {}

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        report.serialize_order(ser);
    }

private:
    ImplementSerializable(SST::SnnDL::TassLfP0ReportEvent)
};

}} // namespace SST::SnnDL

#endif // SNNDL_TASS_LF_P0_REPORT_EVENT_H

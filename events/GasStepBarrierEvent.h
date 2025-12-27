// -*- c++ -*-
//
// GasStepBarrierEvent:
// - 全局 Step/GAS 同步的控制面事件（不携带 Spike/BCSR/权重语义）
// - 用于 GlobalGasStepController <-> MultiCorePE 的 barrier 协议
//

#ifndef SNNDL_GAS_STEP_BARRIER_EVENT_H
#define SNNDL_GAS_STEP_BARRIER_EVENT_H

#include <cstdint>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST { namespace SnnDL {

enum class GasStepBarrierOp : uint8_t {
    Unknown   = 0,
    PeReady   = 1,
    StartStep = 2,
    PeDone    = 3,
};

class GasStepBarrierEvent : public SST::Event {
public:
    uint8_t op = static_cast<uint8_t>(GasStepBarrierOp::Unknown);
    uint32_t seq = 0;
    uint32_t src_node = 0;

    GasStepBarrierEvent() = default;

    GasStepBarrierEvent(GasStepBarrierOp op_, uint32_t seq_, uint32_t src_node_)
        : SST::Event(),
          op(static_cast<uint8_t>(op_)),
          seq(seq_),
          src_node(src_node_)
    {}

    GasStepBarrierOp operation() const { return static_cast<GasStepBarrierOp>(op); }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(op);
        SST_SER(seq);
        SST_SER(src_node);
    }

private:
    ImplementSerializable(SST::SnnDL::GasStepBarrierEvent)
};

}} // namespace SST::SnnDL

#endif // SNNDL_GAS_STEP_BARRIER_EVENT_H


// -*- c++ -*-
// GasCustomCmd.h: Custom control-plane commands for GAS phases

#pragma once

#include <sst/core/interfaces/stdMem.h>
#include <string>

namespace SST { namespace SnnDL {

// GAS control opcodes
enum class GasOp : uint8_t {
    BeginGather = 1,
    EndGather   = 2,
    BeginApply  = 3,
    EndApply    = 4,
    BeginScatter= 5,
    EndScatter  = 6,
    FlushSRAM   = 7,
    SetSlice    = 8,
};

// CustomData payload carried inside StandardMem::CustomReq/Resp
struct GasOpData : public SST::Interfaces::StandardMem::CustomData {
    GasOp op = GasOp::BeginGather;
    uint32_t superstep = 0;
    uint32_t slice = 0;
    uint32_t total_slices = 1;
    bool flag = false; // generic flag (e.g., flush)

    GasOpData() = default;
    GasOpData(GasOp _op, uint32_t ss=0, uint32_t sl=0, uint32_t tot=1, bool fl=false)
        : op(_op), superstep(ss), slice(sl), total_slices(tot), flag(fl) {}

    // CustomData API
    SST::Interfaces::StandardMem::CustomData* makeResponse() override {
        // One-way by default
        return new GasOpData(GasOp::EndScatter, superstep, slice, total_slices, flag);
    }
    bool needsResponse() override { return false; }
    SST::Interfaces::StandardMem::Addr getRoutingAddress() override { return 0; }
    uint64_t getSize() override { return 0; }
    std::string getString() override {
        return std::string("GasOp:") + std::to_string((int)op) +
               ",ss=" + std::to_string(superstep) +
               ",sl=" + std::to_string(slice) +
               ",tot=" + std::to_string(total_slices) +
               ",fl=" + (flag?"1":"0");
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        SST_SER(op);
        SST_SER(superstep);
        SST_SER(slice);
        SST_SER(total_slices);
        SST_SER(flag);
    }
    ImplementSerializable(SST::SnnDL::GasOpData);
};

}} // namespace

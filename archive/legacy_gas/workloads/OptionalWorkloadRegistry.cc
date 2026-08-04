// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "OptionalWorkloadRegistry.h"

#include "ICoreWorkload.h"
#include "workloads/riscv_snn/RiscvSnnWorkload.h"
#include "workloads/stream/StreamWorkload.h"
#include "workloads/tensor/TensorWorkload.h"

namespace SST { namespace SnnDL {

namespace {

std::unique_ptr<ICoreWorkload> makeRiscvSnn_() {
    return std::make_unique<RiscvSnnWorkload>();
}

std::unique_ptr<ICoreWorkload> makeStream_() {
    return std::make_unique<StreamWorkload>();
}

std::unique_ptr<ICoreWorkload> makeTensor_() {
    return std::make_unique<TensorWorkload>();
}

struct BuiltinRegistrar {
    BuiltinRegistrar() {
        registerOptionalWorkload("riscv_snn", &makeRiscvSnn_);
        registerOptionalWorkload("stream", &makeStream_);
        registerOptionalWorkload("tensor", &makeTensor_);
    }
};

BuiltinRegistrar builtin_registrar_;

} // namespace

}} // namespace SST::SnnDL

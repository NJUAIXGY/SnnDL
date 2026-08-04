// Explicit Research registration for the optional RISC-V shadow runtime bridge.
// The default SnnDL entry point intentionally has no dependency on this class.

#include "api/OptionalWorkloadRegistry.h"
#include "api/ISnnAccelRuntimeServices.h"
#include "workloads/riscv_snn/RiscvSnnShadowRuntimeServices.h"

namespace SST { namespace SnnDL {

namespace {

std::unique_ptr<ISnnAccelRuntimeServices> makeRiscvSnnRuntimeBridge_() {
    return std::make_unique<RiscvSnnShadowRuntimeServices>();
}

struct RuntimeBridgeRegistration final {
    RuntimeBridgeRegistration() {
        registerOptionalRuntimeService(
            "riscv_snn_runtime_bridge", &makeRiscvSnnRuntimeBridge_);
    }
};

const RuntimeBridgeRegistration kRuntimeBridgeRegistration{};

} // namespace

}} // namespace SST::SnnDL

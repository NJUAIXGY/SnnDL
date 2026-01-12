// -*- c++ -*-
//
// tests/test_includes.cc: Header include self-check (compile only).
// 目的：在结构重构后尽早暴露“隐式 include / 循环依赖 / 缺失前置声明”等问题。
//
// 约定：
// - 该文件尽量只做包含检查，不引入运行期逻辑。
// - 第一优先：public/control 头应当可被单独包含且不泄露不必要的实现依赖。

// 最关键：控制壳 public 头（应当不依赖 stdMem.h 等重实现头）
#include "control/SnnPESubComponent.h"

// API 层：通用接口（不携带 SNN 语义）
#include "api/IMemoryAccess.h"
#include "api/INocTransport.h"
#include "api/ISpikeTransport.h"
#include "api/IGasCmdSender.h"
#include "api/IGasStageSink.h"
#include "api/IGasStepGate.h"
#include "api/IGlobalStepHooks.h"
#include "api/ISynapseRoute.h"

// 事件类型（上层语义可能存在，但应当可被独立包含）
#include "events/SpikeEvent.h"
#include "events/NocPacketEvent.h"

// 子系统 public 头（应当通过 API 交互）
#include "services/synapse/stdmem/StdMemEndpoint.h"
#include "services/synapse/weights/WeightMemorySubsystem.h"
#include "services/synapse/route/SpikeCommSubsystem.h"
#include "services/synapse/route/SynapseRouteSubsystem.h"
#include "services/noc/NocSubsystem.h"
#include "services/stimulus/StepActivationSubsystem.h"
// Phase: native multicast lab (traffic-only workload)
#include "services/workload/traffic/TrafficWorkload.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // compile-only reference for new kinds/helpers
    (void)SST::SnnDL::NocPacketKind::SpikeKey;

    return 0;
}

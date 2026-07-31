// -*- c++ -*-
//
// tests/test_includes.cc: Header include self-check (compile only).
// 目的：在结构重构后尽早暴露“隐式 include / 循环依赖 / 缺失前置声明”等问题。
//
// 约定：
// - 该文件尽量只做包含检查，不引入运行期逻辑。
// - 第一优先：public/control 头应当可被单独包含且不泄露不必要的实现依赖。

// 最关键：控制壳 public 头（应当不依赖 stdMem.h 等重实现头）
#include "platform/core/SnnPESubComponent.h"

// API 层：通用接口（不携带 SNN 语义）
#include "api/IMemoryAccess.h"
#include "api/INocTransport.h"
#include "api/ISpikeTransport.h"
#include "api/IGasCmdSender.h"
#include "api/IGasStageSink.h"
#include "api/IGasStepGate.h"
#include "api/IGlobalStepHooks.h"
#include "api/ILocalStorageProvider.h"
#include "api/IPeWeightObjectPlaneProvider.h"
#include "api/ISynapseRoute.h"
#include "api/IDmaTaggedAccess.h"
#include "api/IDmaSchedulerProvider.h"

// 事件类型（上层语义可能存在，但应当可被独立包含）
#include "events/SpikeEvent.h"
#include "events/NocPacketEvent.h"

// 子系统 public 头（应当通过 API 交互）
#include "snn/synapse/stdmem/StdMemEndpoint.h"
#include "snn/synapse/weights/WeightMemorySubsystem.h"
#include "snn/synapse/route/SpikeCommSubsystem.h"
#include "snn/synapse/route/SynapseRouteSubsystem.h"
#include "research/route3d/SynapseRouteSubsystem3D.h"
#include "snn/synapse/route/SpikeTileNocCodec.h"
#include "research/noc3d/MulticastRouter3DNative.h"
#include "platform/noc/NocSubsystem.h"
#include "snn/stimulus/StepActivationSubsystem.h"
#include "platform/memory/PeDmaScheduler.h"
#include "platform/memory/DmaMemAccessProxy.h"
#include "research/local_storage/LocalStorageTypes.h"
#include "research/local_storage/LocalStorageHierarchyController.h"
#include "research/local_storage/PeInternalPodShadowGate.h"
#include "research/local_storage/PeWeightObjectPlane.h"
// Phase: native multicast lab (traffic-only workload)
#include "workloads/traffic/TrafficWorkload.h"
// Phase: tensor workload (systolic microbench)
#include "workloads/tensor/TensorWorkload.h"
// Phase: riscv_snn workload skeleton + shared backend contract
#include "workloads/common/SnnAccelBackend.h"
#include "workloads/riscv_snn/RiscvSnnAbi.h"
#include "workloads/riscv_snn/RiscvSnnAsm.h"
#include "workloads/riscv_snn/RiscvSnnBootDriver.h"
#include "workloads/riscv_snn/RiscvSnnElfWriter.h"
#include "workloads/riscv_snn/RiscvSnnFirmwareLoader.h"
#include "workloads/riscv_snn/RiscvSnnHart.h"
#include "workloads/riscv_snn/RiscvSnnIss.h"
#include "workloads/riscv_snn/RiscvSnnMemoryImage.h"
#include "workloads/riscv_snn/RiscvSnnQueueContract.h"
#include "workloads/riscv_snn/RiscvSnnSampleFirmware.h"
#include "workloads/riscv_snn/RiscvSnnWorkload.h"
// Workload stats registry (PE aggregation)
#include "platform/stats/WorkloadStatsRegistry.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    // compile-only reference for new kinds/helpers
    (void)SST::SnnDL::NocPacketKind::SpikeKey;
    (void)SST::SnnDL::NocPacketKind::SpikeTileKey;
    SST::SnnDL::SpikeTileNocCodec::WireSpikeTileKeyV1 tile_ws;
    (void)tile_ws;

    SST::SnnDL::TensorWorkload::Config tensor_cfg;
    tensor_cfg.collective_type = "allreduce";
    tensor_cfg.collective_bytes = 1024;
    tensor_cfg.collective_period_cycles = 100;
    tensor_cfg.collective_pattern = "ring";
    tensor_cfg.collective_packet_bytes = 256;
    tensor_cfg.exec_mode = "tile";
    tensor_cfg.tile_schedule = "mkn";
    tensor_cfg.writeback_policy = "at_end_of_k";
    tensor_cfg.collective_blocking = true;
    tensor_cfg.collective_scope = "per_core";

    SST::SnnDL::RiscvSnnHart riscv_hart;
    riscv_hart.reset(0x1000);
    SST::SnnDL::RiscvSnnIss riscv_iss;
    SST::SnnDL::RiscvSnnBootDriver riscv_boot_driver;
    riscv_boot_driver.configureExternalFirmware();
    SST::SnnDL::riscv_snn::RiscvSnnMemoryImage riscv_image;
    SST::SnnDL::riscv_snn::RiscvSnnQueueContract riscv_queues;
    SST::SnnDL::riscv_snn::RiscvSnnSampleMetadata riscv_sample_metadata;
    SST::SnnDL::riscv_snn::RiscvSnnSampleProgram riscv_sample_program;
    (void)riscv_queues.configure(8, 8, 4);
    auto riscv_backend = SST::SnnDL::makeNullSnnAccelBackend();
    auto riscv_workload = std::make_unique<SST::SnnDL::RiscvSnnWorkload>();
    SST::SnnDL::riscv_snn::RiscvSnnElfLoadSegment riscv_load_segment;
    (void)riscv_hart;
    (void)riscv_iss;
    (void)riscv_boot_driver;
    (void)riscv_image;
    (void)riscv_queues;
    (void)riscv_sample_metadata;
    (void)riscv_sample_program;
    (void)riscv_backend;
    (void)riscv_workload;
    (void)riscv_load_segment;
    (void)SST::SnnDL::riscv_snn::kCsrMsnnCfg;
    (void)SST::SnnDL::riscv_snn::asmv1::encodeEbreak();
    (void)SST::SnnDL::riscv_snn::canonicalSampleFirmwareProgramNames();
    (void)SST::SnnDL::riscv_snn::sampleFirmwareMetadata();
    (void)SST::SnnDL::riscv_snn::sampleFirmwareProgramNames();

    return 0;
}

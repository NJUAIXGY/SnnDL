AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I$(srcdir) \
	-I$(srcdir)/api \
	-I$(srcdir)/events \
	-I$(srcdir)/components \
	-I$(srcdir)/platform/core \
	-I$(srcdir)/snn/compute \
	-I$(srcdir)/snn/profiling \
	-I$(SST_PREFIX)/include

# 可选的MPI支持 - 不影响现有构建
# 暂时注释掉，等待configure.ac配置完成
# if SST_CONFIG_HAVE_MPI
# AM_CPPFLAGS += $(MPI_CPPFLAGS)
# endif

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la
libSnnDL_la_SOURCES = \
	api/CoreShellAPI.h \
	api/SnnInterface.h \
	api/ISpikeTransport.h \
	api/INocTransport.h \
	api/GasOps.h \
	api/IGasCmdSender.h \
	api/IGasStageSink.h \
	api/IGasStepGate.h \
	api/IGlobalStepHooks.h \
	api/ICoreMemoryLink.h \
	api/IMemoryAccess.h \
	api/IDmaTaggedAccess.h \
	api/IDmaSchedulerProvider.h \
	api/IPeSharedCoreFabricProvider.h \
	api/ILocalStorageProvider.h \
	api/ICoreWorkload.h \
	api/ISnnAccelRuntimeServices.h \
	api/ISpikeWorkload.h \
	api/WorkloadConfig.h \
	api/CoreWorkloadFactory.h \
	api/ISynapseRoute.h \
	api/SynapseRouteBuildConfig.h \
	api/SnnWeightReader.h \
	api/IWeightReaderAdopter.h \
	api/SnnDLLogging.h \
	events/GatingDecisionEvent.h \
	events/GasStepBarrierEvent.h \
	events/SimpleTestEvent.cc \
	events/SimpleTestEvent.h \
	events/SpikeEvent.h \
	events/SpikeEventWrapper.cc \
	events/SpikeEventWrapper.h \
	events/SpikePacket.h \
	components/gas/GlobalGasStepController.cc \
	components/gas/GlobalGasStepController.h \
	components/CachelineFragmentMemIF.cc \
	components/CachelineFragmentMemIF.h \
	components/GatherBufferIF.cc \
	components/GatherBufferIF.h \
	components/gather/GatherBufferIFConfig.cc \
	components/gather/GatherBufferIFConfig.h \
	components/gather/apply/DramAwareModel.cc \
	components/gather/apply/DramAwareModel.h \
	components/GatingPE.cc \
	components/GatingPE.h \
	components/MemKCalBench.cc \
	components/MemKCalBench.h \
	components/multicore/MultiCorePEConfig.cc \
	components/multicore/MultiCorePEConfig.h \
	components/multicore/MultiCorePEObservability.cc \
	components/MultiCorePE.cc \
	components/MultiCorePE.h \
	components/SnnNIC.cc \
	components/SnnNIC.h \
	components/stimulus/SpikeSource.cc \
	components/stimulus/SpikeSource.h \
	components/WeightLoader.cc \
	components/WeightLoader.h \
	platform/stats/IWorkloadStatsModule.h \
	platform/stats/KeyedCounterStatsModule.h \
	platform/stats/StreamWorkloadStatsModule.h \
	platform/stats/TensorWorkloadStatsModule.h \
	platform/stats/WorkloadStatsRegistry.cc \
	platform/stats/WorkloadStatsRegistry.h \
	platform/core/SnnPESubComponent.cc \
	platform/core/SnnPESubComponentStats.cc \
	platform/core/SnnPESubComponent.h \
	platform/core/SnnPESubComponent_impl.h \
	platform/core/SnnPESubComponent_bcsr.cc \
	platform/core/SnnPESubComponent_routing.cc \
	platform/core/SnnPEApplyScatter.cc \
	platform/core/SnnPEOrchestrators.cc \
	snn/compute/SnnComputeCore.cc \
	snn/compute/SnnComputeCore.h \
	snn/compute/SnnCoreEngine.cc \
	snn/compute/SnnCoreEngine.h \
	snn/compute/SnnLearningCore.cc \
	snn/compute/SnnLearningCore.h \
	snn/compute/IWeightAwareComputeCore.h \
	snn/compute/SnnNeuronModel.h \
	snn/compute/SynapseManager.h \
	snn/synapse/gas/AccumulatorOps.cc \
	snn/synapse/gas/AccumulatorOps.h \
	snn/synapse/gas/GasCustomCmd.h \
	snn/synapse/gas/GasEdgeCollector.cc \
	snn/synapse/gas/GasEdgeCollector.h \
	snn/synapse/gas/GasPhaseController.cc \
	snn/synapse/gas/GasPhaseController.h \
	snn/synapse/stdmem/StdMemEndpoint.cc \
	snn/synapse/stdmem/StdMemEndpoint.h \
	snn/synapse/stdmem/SnnPESubComponent_mem.cc \
	platform/noc/OptimizedInternalRing.cc \
	platform/noc/OptimizedInternalRing.h \
	components/noc/SimpleNetworkWrapper.cc \
	components/noc/SimpleNetworkWrapper.h \
	snn/synapse/weights/SnnBcsrWeightManager.cc \
	snn/synapse/weights/SnnBcsrWeightManager.h \
	snn/synapse/weights/SnnWeightDiagnostics.cc \
	snn/synapse/weights/SnnWeightDiagnostics.h \
	components/noc/SnnNetworkAdapter.cc \
	components/noc/SnnNetworkAdapter.h \
	components/noc/MulticastRouter.cc \
	components/noc/MulticastRouter.h \
	research/noc3d/MulticastRouter3DNative.cc \
	research/noc3d/MulticastRouter3DNative.h \
	research/noc3d/MulticastRouter3DNativeConfig.h \
	research/noc3d/Noc3DSmokeSource.cc \
	research/noc3d/Noc3DSmokeSource.h \
	research/noc3d/Noc3DSmokeSink.cc \
	research/noc3d/Noc3DSmokeSink.h \
	research/noc3d/HBMStackStub.cc \
	research/noc3d/HBMStackStub.h \
	components/noc/MulticastNIC.cc \
	components/noc/MulticastNIC.h \
	snn/profiling/SnnProfiler.h \
	snn/synapse/route/SnnRouteProvider.cc \
	snn/synapse/route/SnnRouteProvider.h \
	snn/synapse/route/SpikeNocCodec.cc \
	snn/synapse/route/SpikeNocCodec.h \
	snn/synapse/route/BcsrRouteBuilder.cc \
	snn/synapse/route/BcsrRouteBuilder.h \
	snn/synapse/route/StepBcsrReachability.cc \
	snn/synapse/route/StepBcsrReachability.h \
	snn/synapse/route/SynapseRouteSubsystem.cc \
	snn/synapse/route/SynapseRouteSubsystem.h \
	research/route3d/Route3DNodeMapper.h \
	research/route3d/SynapseRouteSubsystem3D.cc \
	research/route3d/SynapseRouteSubsystem3D.h \
	snn/stimulus/StepActivationSubsystem.cc \
	snn/stimulus/StepActivationSubsystem.h \
	snn/stimulus/ExternalSpikeInputSubsystem.cc \
	snn/stimulus/ExternalSpikeInputSubsystem.h \
	platform/noc/NocSubsystem.cc \
	platform/noc/NocSubsystem.h \
	snn/synapse/weights/WeightAccessor.cc \
	snn/synapse/weights/WeightAccessor.h \
	snn/synapse/weights/WeightCacheOps.cc \
	snn/synapse/weights/WeightCacheOps.h \
	snn/synapse/weights/WeightMemorySubsystem.cc \
	snn/synapse/weights/WeightMemorySubsystemVerification.cc \
	snn/synapse/weights/WeightMemorySubsystem.h \
	snn/synapse/route/SpikeCommSubsystem.cc \
	snn/synapse/route/SpikeCommSubsystem.h \
	snn/synapse/route/SpikePacketTransport.h \
	snn/synapse/route/SpikePacketBridge.cc \
	snn/synapse/route/SpikePacketBridge.h \
	workloads/CoreWorkloadFactory.cc \
	workloads/common/SnnAccelBackend.cc \
	workloads/common/SnnAccelBackend.h \
	workloads/common/SnnAccelBackendContract.h \
	workloads/layout/NormalizedNeuronLayout.cc \
	workloads/layout/NormalizedNeuronLayout.h \
	workloads/riscv_snn/RiscvSnnAbi.h \
	workloads/riscv_snn/RiscvSnnAsm.h \
	workloads/riscv_snn/RiscvSnnBootDriver.cc \
	workloads/riscv_snn/RiscvSnnBootDriver.h \
	workloads/riscv_snn/RiscvSnnElfWriter.h \
	workloads/riscv_snn/RiscvSnnFirmwareLoader.cc \
	workloads/riscv_snn/RiscvSnnFirmwareLoader.h \
	workloads/riscv_snn/RiscvSnnHart.cc \
	workloads/riscv_snn/RiscvSnnHart.h \
	workloads/riscv_snn/RiscvSnnIss.cc \
	workloads/riscv_snn/RiscvSnnIss.h \
	workloads/riscv_snn/RiscvSnnMemoryImage.h \
	workloads/riscv_snn/RiscvSnnQueueContract.h \
	workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
	workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.h \
	workloads/riscv_snn/RiscvSnnSampleFirmware.h \
	workloads/riscv_snn/RiscvSnnShadowTransportExport.cc \
	workloads/riscv_snn/RiscvSnnShadowTransportExport.h \
	workloads/riscv_snn/RiscvSnnShadowRuntimeServices.cc \
	workloads/riscv_snn/RiscvSnnShadowRuntimeServices.h \
	workloads/riscv_snn/RiscvSnnWorkload.cc \
	workloads/riscv_snn/RiscvSnnWorkload.h \
	workloads/snn/SnnWorkload.cc \
	workloads/snn/SnnWorkload.h \
	workloads/stream/StreamWorkload.cc \
	workloads/stream/StreamWorkload.h \
	workloads/traffic/TrafficWorkload.cc \
	workloads/traffic/TrafficWorkload.h \
	workloads/traffic_mem/TrafficMemWorkload.cc \
	workloads/traffic_mem/TrafficMemWorkload.h \
	workloads/tensor/TensorWorkload.cc \
	workloads/tensor/TensorWorkloadCollective.cc \
	workloads/tensor/TensorWorkloadConfig.cc \
	workloads/tensor/TensorWorkloadProgram.cc \
	workloads/tensor/TensorWorkloadStats.cc \
	workloads/tensor/TensorWorkload.h \
	platform/memory/sram_sim/layout/VirtualSramLayout.h \
	platform/memory/sram_sim/model/BankedSramModel.cc \
	platform/memory/sram_sim/model/BankedSramModel.h \
	research/local_storage/LocalStorageTypes.h \
	research/local_storage/LocalStorageHierarchyController.cc \
	research/local_storage/LocalStorageHierarchyController.h \
	research/local_storage/PeWeightObjectPlane.cc \
	research/local_storage/PeWeightObjectPlane.h \
	platform/memory/DmaMemAccessProxy.cc \
	platform/memory/DmaMemAccessProxy.h \
	research/pe_fabric/PulseAgendaScorer.cc \
	research/pe_fabric/PulseAgendaScorer.h \
	research/pe_fabric/PulseDescriptor.h \
	research/pe_fabric/PeSharedCoreFabric.cc \
	research/pe_fabric/PeSharedCoreFabric.h \
	platform/memory/PeDmaScheduler.cc \
	platform/memory/PeDmaScheduler.h \
	platform/memory/StandardMemAccess.cc \
	platform/memory/StandardMemAccess.h

#
# Note: sources above include the production runtime and explicitly selected research modules.
#

libSnnDL_la_LDFLAGS = -module -avoid-version
RISCV_SNN_TEST_LINK = $(CXXLD) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) $(CXXFLAGS)

# 可选的MPI链接库 - 暂时注释掉
# if SST_CONFIG_HAVE_MPI
# libSnnDL_la_LDFLAGS += $(MPI_LDFLAGS)
# libSnnDL_la_LIBADD = $(MPI_LIBS)
# endif

.PHONY: test-compile test-riscv-snn-protocols test-riscv-snn-backend-timing-contract test-riscv-snn-runtime-service-provider test-riscv-snn-runtime-bridge-backend test-riscv-snn-firmware-protocol test-riscv-snn-toolchain-firmware-protocol test-riscv-snn-stimulus-gating
test-riscv-snn-protocols: test-riscv-snn-backend-timing-contract test-riscv-snn-runtime-service-provider test-riscv-snn-runtime-bridge-backend test-riscv-snn-firmware-protocol test-riscv-snn-toolchain-firmware-protocol test-riscv-snn-stimulus-gating

test-riscv-snn-stimulus-gating:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_stimulus_gating.cc \
		-o $(builddir)/test_riscv_snn_stimulus_gating$(EXEEXT) && \
		$(builddir)/test_riscv_snn_stimulus_gating$(EXEEXT) && \
		rg -q "ex_rt\\.enabled = workloadAllowsSnnStimulus\\(cfg\\.workload_kind\\);" $(srcdir)/components/MultiCorePE.cc && \
		rg -q "if \\(!rt_\\.enabled\\)" $(srcdir)/snn/stimulus/ExternalSpikeInputSubsystem.cc

test-riscv-snn-backend-timing-contract:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_backend_timing_contract.cc \
		$(srcdir)/workloads/common/SnnAccelBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnHart.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnIss.cc \
		-o $(builddir)/test_riscv_snn_backend_timing_contract$(EXEEXT) && \
		$(builddir)/test_riscv_snn_backend_timing_contract$(EXEEXT)

test-riscv-snn-runtime-service-provider:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_runtime_service_provider.cc \
		$(srcdir)/workloads/common/SnnAccelBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnShadowTransportExport.cc \
		-o $(builddir)/test_riscv_snn_runtime_service_provider$(EXEEXT) && \
		$(builddir)/test_riscv_snn_runtime_service_provider$(EXEEXT)

test-riscv-snn-runtime-bridge-backend:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_runtime_bridge_backend.cc \
		$(srcdir)/workloads/common/SnnAccelBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
		-o $(builddir)/test_riscv_snn_runtime_bridge_backend$(EXEEXT) && \
		$(builddir)/test_riscv_snn_runtime_bridge_backend$(EXEEXT)

test-riscv-snn-firmware-protocol:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_firmware_protocol.cc \
		$(srcdir)/workloads/common/SnnAccelBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnHart.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnIss.cc \
		-o $(builddir)/test_riscv_snn_firmware_protocol$(EXEEXT) && \
		$(builddir)/test_riscv_snn_firmware_protocol$(EXEEXT)

test-riscv-snn-toolchain-firmware-protocol:
	$(RISCV_SNN_TEST_LINK) $(srcdir)/tests/test_riscv_snn_toolchain_firmware_protocol.cc \
		$(srcdir)/workloads/common/SnnAccelBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnRuntimeBridgeBackend.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnHart.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnIss.cc \
		$(srcdir)/workloads/riscv_snn/RiscvSnnFirmwareLoader.cc \
		-o $(builddir)/test_riscv_snn_toolchain_firmware_protocol$(EXEEXT) && \
		$(builddir)/test_riscv_snn_toolchain_firmware_protocol$(EXEEXT)

test-compile:
	$(CXXCOMPILE) -c $(srcdir)/tests/test_includes.cc -o $(builddir)/test_includes.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_pe_internal_pod_shadow_gate.cc -o $(builddir)/test_pe_internal_pod_shadow_gate.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_pe_internal_pod_shadow_wms_seam.cc -o $(builddir)/test_pe_internal_pod_shadow_wms_seam.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_abi.cc -o $(builddir)/test_riscv_snn_abi.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_backend_timing_contract.cc -o $(builddir)/test_riscv_snn_backend_timing_contract.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_boot_driver.cc -o $(builddir)/test_riscv_snn_boot_driver.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_firmware_loader.cc -o $(builddir)/test_riscv_snn_firmware_loader.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_firmware_tools.cc -o $(builddir)/test_riscv_snn_firmware_tools.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_firmware_protocol.cc -o $(builddir)/test_riscv_snn_firmware_protocol.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_toolchain_firmware_protocol.cc -o $(builddir)/test_riscv_snn_toolchain_firmware_protocol.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_iss.cc -o $(builddir)/test_riscv_snn_iss.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_queue_contract.cc -o $(builddir)/test_riscv_snn_queue_contract.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_runtime_bridge_backend.cc -o $(builddir)/test_riscv_snn_runtime_bridge_backend.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_runtime_service_provider.cc -o $(builddir)/test_riscv_snn_runtime_service_provider.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_riscv_snn_stimulus_gating.cc -o $(builddir)/test_riscv_snn_stimulus_gating.o

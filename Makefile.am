AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I$(srcdir) \
	-I$(srcdir)/api \
	-I$(srcdir)/events \
	-I$(srcdir)/components \
	-I$(srcdir)/control \
	-I$(srcdir)/compute \
	-I$(srcdir)/services \
	-I$(SST_PREFIX)/include

# 可选的MPI支持 - 不影响现有构建
# 暂时注释掉，等待configure.ac配置完成
# if SST_CONFIG_HAVE_MPI
# AM_CPPFLAGS += $(MPI_CPPFLAGS)
# endif

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la
libSnnDL_la_SOURCES = \
	api/SnnCoreAPI.h \
	api/CoreShellAPI.h \
	api/SnnInterface.h \
	api/SnnPEParentInterface.h \
	api/ISpikeTransport.h \
	api/INocTransport.h \
	api/IManualWindowDrive.h \
	api/GasOps.h \
	api/IGasCmdSender.h \
	api/IGasStageSink.h \
	api/IGasStepGate.h \
	api/IGlobalStepHooks.h \
	api/ICoreMemoryLink.h \
	api/IMemoryAccess.h \
	api/ICoreWorkload.h \
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
	components/gather/apply/DramAwareTuner.cc \
	components/gather/apply/DramAwareTuner.h \
	components/GatingPE.cc \
	components/GatingPE.h \
	components/MemKCalBench.cc \
	components/MemKCalBench.h \
	components/multicore/MultiCorePEConfig.cc \
	components/multicore/MultiCorePEConfig.h \
	components/MultiCorePE.cc \
	components/MultiCorePE.h \
	components/SnnNIC.cc \
	components/SnnNIC.h \
	components/SnnPE.cc \
	components/SnnPE.h \
	components/stimulus/SpikeSource.cc \
	components/stimulus/SpikeSource.h \
	components/WeightLoader.cc \
	components/WeightLoader.h \
	components/workload_stats/IWorkloadStatsModule.h \
	components/workload_stats/KeyedCounterStatsModule.h \
	components/workload_stats/StreamWorkloadStatsModule.h \
	components/workload_stats/TensorWorkloadStatsModule.h \
	components/workload_stats/WorkloadStatsRegistry.cc \
	components/workload_stats/WorkloadStatsRegistry.h \
	control/SnnPESubComponent.cc \
	control/SnnPESubComponent.h \
	control/SnnPESubComponent_impl.h \
	control/SnnPESubComponent_bcsr.cc \
	control/SnnPESubComponent_routing.cc \
	control/SnnPESubComponent_scheme1.cc \
	control/SnnPESubComponent_spike.cc \
	control/SnnPEApplyScatter.cc \
	control/SnnPEOrchestrators.cc \
	compute/SnnComputeCore.cc \
	compute/SnnComputeCore.h \
	compute/SnnCoreEngine.cc \
	compute/SnnCoreEngine.h \
	compute/SnnLearningCore.cc \
	compute/SnnLearningCore.h \
	compute/IWeightAwareComputeCore.h \
	compute/SnnNeuronModel.h \
	compute/SynapseManager.h \
	services/synapse/gas/AccumulatorOps.cc \
	services/synapse/gas/AccumulatorOps.h \
	services/synapse/gas/GasCustomCmd.h \
	services/synapse/gas/GasEdgeCollector.cc \
	services/synapse/gas/GasEdgeCollector.h \
	services/synapse/gas/GasPhaseController.cc \
	services/synapse/gas/GasPhaseController.h \
	services/synapse/stdmem/StdMemEndpoint.cc \
	services/synapse/stdmem/StdMemEndpoint.h \
	services/synapse/stdmem/SnnPESubComponent_mem.cc \
	services/noc/OptimizedInternalRing.cc \
	services/noc/OptimizedInternalRing.h \
	components/noc/SimpleNetworkWrapper.cc \
	components/noc/SimpleNetworkWrapper.h \
	services/synapse/weights/SnnBcsrWeightManager.cc \
	services/synapse/weights/SnnBcsrWeightManager.h \
	services/synapse/weights/SnnWeightDiagnostics.cc \
	services/synapse/weights/SnnWeightDiagnostics.h \
	components/noc/SnnNetworkAdapter.cc \
	components/noc/SnnNetworkAdapter.h \
	components/noc/MulticastRouter.cc \
	components/noc/MulticastRouter.h \
	components/noc/MulticastNIC.cc \
	components/noc/MulticastNIC.h \
	services/SnnProfiler.h \
	services/synapse/route/SnnRouteProvider.cc \
	services/synapse/route/SnnRouteProvider.h \
	services/synapse/route/BcsrRouteBuilder.cc \
	services/synapse/route/BcsrRouteBuilder.h \
	services/synapse/route/StepBcsrReachability.cc \
	services/synapse/route/StepBcsrReachability.h \
	services/synapse/route/SynapseRouteSubsystem.cc \
	services/synapse/route/SynapseRouteSubsystem.h \
	services/stimulus/StepActivationSubsystem.cc \
	services/stimulus/StepActivationSubsystem.h \
	services/stimulus/ExternalSpikeInputSubsystem.cc \
	services/stimulus/ExternalSpikeInputSubsystem.h \
	services/noc/NocSubsystem.cc \
	services/noc/NocSubsystem.h \
	services/synapse/weights/WeightAccessor.cc \
	services/synapse/weights/WeightAccessor.h \
	services/synapse/weights/WeightCacheOps.cc \
	services/synapse/weights/WeightCacheOps.h \
	services/synapse/weights/WeightMemorySubsystem.cc \
	services/synapse/weights/WeightMemorySubsystem.h \
	services/synapse/route/SpikeCommSubsystem.cc \
	services/synapse/route/SpikeCommSubsystem.h \
	services/synapse/route/SpikePacketTransport.h \
	services/synapse/route/SpikePacketBridge.cc \
	services/synapse/route/SpikePacketBridge.h \
	services/workload/CoreWorkloadFactory.cc \
	services/workload/layout/NormalizedNeuronLayout.cc \
	services/workload/layout/NormalizedNeuronLayout.h \
	services/workload/snn/SnnWorkload.cc \
	services/workload/snn/SnnWorkload.h \
	services/workload/stream/StreamWorkload.cc \
	services/workload/stream/StreamWorkload.h \
	services/workload/traffic/TrafficWorkload.cc \
	services/workload/traffic/TrafficWorkload.h \
	services/workload/tensor/TensorWorkload.cc \
	services/workload/tensor/TensorWorkload.h \
	services/memory/StandardMemAccess.cc \
	services/memory/StandardMemAccess.h

#
# Note: sources above already include GatherBufferIF / MemKCalBench / MPI files.
#

libSnnDL_la_LDFLAGS = -module -avoid-version

# 可选的MPI链接库 - 暂时注释掉
# if SST_CONFIG_HAVE_MPI
# libSnnDL_la_LDFLAGS += $(MPI_LDFLAGS)
# libSnnDL_la_LIBADD = $(MPI_LIBS)
# endif

.PHONY: test-compile
test-compile:
	$(CXXCOMPILE) -c $(srcdir)/tests/test_includes.cc -o $(builddir)/test_includes.o

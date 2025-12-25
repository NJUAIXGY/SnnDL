AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I$(srcdir) \
	-I$(srcdir)/api \
	-I$(srcdir)/events \
	-I$(srcdir)/components \
	-I$(srcdir)/components/mpi \
	-I$(srcdir)/control \
	-I$(srcdir)/compute \
	-I$(srcdir)/services \
	-I/home/xgy/remote/sst_install/include

# 可选的MPI支持 - 不影响现有构建
# 暂时注释掉，等待configure.ac配置完成
# if SST_CONFIG_HAVE_MPI
# AM_CPPFLAGS += $(MPI_CPPFLAGS)
# endif

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la
libSnnDL_la_SOURCES = \
	api/SnnCoreAPI.h \
	api/SnnInterface.h \
	api/SnnPEParentInterface.h \
	api/ISpikeTransport.h \
	api/INocTransport.h \
	api/NocSpikeTransport.h \
	api/IManualWindowDrive.h \
	api/IMemoryAccess.h \
	api/ISynapseRoute.h \
	api/SynapseRouteBuildConfig.h \
	api/SnnWeightReader.h \
	events/GatingDecisionEvent.h \
	events/SimpleTestEvent.cc \
	events/SimpleTestEvent.h \
	events/SpikeEvent.h \
	events/SpikeEventWrapper.cc \
	events/SpikeEventWrapper.h \
	events/SpikePacket.h \
	components/GatherBufferIF.cc \
	components/GatherBufferIF.h \
	components/GatingPE.cc \
	components/GatingPE.h \
	components/MemKCalBench.cc \
	components/MemKCalBench.h \
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
	components/mpi/MPIMultiCorePE.cc \
	components/mpi/MPIMultiCorePE.h \
	components/mpi/MPITypes.cc \
	components/mpi/MPITypes.h \
	control/SnnPESubComponent.cc \
	control/SnnPESubComponent.h \
	control/SnnPESubComponent_bcsr.cc \
	control/SnnPESubComponent_mem.cc \
	control/SnnPESubComponent_routing.cc \
	control/SnnPESubComponent_scheme1.cc \
	control/SnnPESubComponent_spike.cc \
	control/SnnPEApplyScatter.cc \
	control/SnnPEOrchestrators.cc \
	control/StageEventHub.cc \
	control/StageEventHub.h \
	compute/SnnComputeCore.cc \
	compute/SnnComputeCore.h \
	compute/SnnCoreEngine.cc \
	compute/SnnCoreEngine.h \
	compute/SnnLearningCore.cc \
	compute/SnnLearningCore.h \
	compute/SnnNeuronModel.h \
	compute/SynapseManager.h \
	services/synapse/gas/AccumulatorOps.cc \
	services/synapse/gas/AccumulatorOps.h \
	services/synapse/gas/GasCustomCmd.h \
	services/synapse/gas/GasEdgeCollector.cc \
	services/synapse/gas/GasEdgeCollector.h \
	services/synapse/gas/GasPhaseController.cc \
	services/synapse/gas/GasPhaseController.h \
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

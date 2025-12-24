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
	components/SpikeSource.cc \
	components/SpikeSource.h \
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
	compute/SnnWeightDiagnostics.cc \
	compute/SnnWeightDiagnostics.h \
	compute/SynapseManager.h \
	services/AccumulatorOps.cc \
	services/AccumulatorOps.h \
	services/GasCustomCmd.h \
	services/GasEdgeCollector.cc \
	services/GasEdgeCollector.h \
	services/GasPhaseController.cc \
	services/GasPhaseController.h \
	services/OptimizedInternalRing.cc \
	services/OptimizedInternalRing.h \
	services/SimpleNetworkWrapper.cc \
	services/SimpleNetworkWrapper.h \
	services/SnnBcsrWeightManager.cc \
	services/SnnBcsrWeightManager.h \
	services/SnnNetworkAdapter.cc \
	services/SnnNetworkAdapter.h \
	services/SnnProfiler.h \
	services/SnnRouteProvider.cc \
	services/SnnRouteProvider.h \
	services/SynapseRouteSubsystem.cc \
	services/SynapseRouteSubsystem.h \
	services/StepActivationSubsystem.cc \
	services/StepActivationSubsystem.h \
	services/NocSubsystem.cc \
	services/NocSubsystem.h \
	services/WeightAccessor.cc \
	services/WeightAccessor.h \
	services/WeightCacheOps.cc \
	services/WeightCacheOps.h \
	services/WeightMemorySubsystem.cc \
	services/WeightMemorySubsystem.h \
	services/SpikeCommSubsystem.cc \
	services/SpikeCommSubsystem.h \
	services/StandardMemAccess.cc \
	services/StandardMemAccess.h \
	services/StandardMemBackend.cc \
	services/StandardMemBackend.h

#
# Note: sources above already include GatherBufferIF / MemKCalBench / MPI files.
#

libSnnDL_la_LDFLAGS = -module -avoid-version

# 可选的MPI链接库 - 暂时注释掉
# if SST_CONFIG_HAVE_MPI
# libSnnDL_la_LDFLAGS += $(MPI_LDFLAGS)
# libSnnDL_la_LIBADD = $(MPI_LIBS)
# endif

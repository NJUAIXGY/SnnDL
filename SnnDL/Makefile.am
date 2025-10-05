AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I/home/anarchy/sst_install/include

# 可选的MPI支持 - 不影响现有构建
# 暂时注释掉，等待configure.ac配置完成
# if SST_CONFIG_HAVE_MPI
# AM_CPPFLAGS += $(MPI_CPPFLAGS)
# endif

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la
libSnnDL_la_SOURCES = \
	SnnPE.cc \
	SnnPE.h \
	SnnCoreAPI.h \
	SnnPESubComponent.cc \
	SnnPESubComponent.h \
	SpikeSource.cc \
	SpikeSource.h \
	SnnNIC.cc \
	SnnNIC.h \
	SnnNetworkAdapter.cc \
	SnnNetworkAdapter.h \
	SimpleNetworkWrapper.cc \
	SimpleNetworkWrapper.h \
	MultiCorePE.cc \
	MultiCorePE.h \
	MultiCorePERouterInterface.cc \
	MultiCorePERouterInterface.h \
	OptimizedInternalRing.cc \
	OptimizedInternalRing.h \
	SnnInterface.h \
	SnnPEParentInterface.h \
	SpikeEvent.h \
	SpikeEventWrapper.h \
	SpikeEventWrapper.cc \
	SimpleTestEvent.h \
	SimpleTestEvent.cc \
	WeightLoader.cc \
	WeightLoader.h

# MPI扩展源文件 - 重新启用测试编译
libSnnDL_la_SOURCES += \
	MPITypes.h \
	MPITypes.cc \
	MPIMultiCorePE.h \
	MPIMultiCorePE.cc

libSnnDL_la_LDFLAGS = -module -avoid-version

# 可选的MPI链接库 - 暂时注释掉
# if SST_CONFIG_HAVE_MPI
# libSnnDL_la_LDFLAGS += $(MPI_LDFLAGS)
# libSnnDL_la_LIBADD = $(MPI_LIBS)
# endif

AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I/home/anarchy/sst_install/include

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

libSnnDL_la_LDFLAGS = -module -avoid-version


## -*- Makefile -*-
##
## SnnDL - Spiking Neural Network Deep Learning Framework for SST
## 

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la

libSnnDL_la_SOURCES = \
    src/sst/elements/SnnDL/SnnPE.h \
    src/sst/elements/SnnDL/SnnPE.cc \
    src/sst/elements/SnnDL/SpikeSource.h \
    src/sst/elements/SnnDL/SpikeSource.cc \
    src/sst/elements/SnnDL/SpikeEvent.h \
    src/sst/elements/SnnDL/SnnInterface.h \
    src/sst/elements/SnnDL/SnnNIC.h \
    src/sst/elements/SnnDL/SnnNIC.cc

libSnnDL_la_CPPFLAGS = \
    $(AM_CPPFLAGS) \
    $(SST_CORE_CPPFLAGS)

libSnnDL_la_LDFLAGS = \
    -module -avoid-version \
    $(SST_CORE_LDFLAGS)

libSnnDL_la_LIBADD = \
    $(SST_CORE_LIBS)

# Install headers
nobase_include_HEADERS = \
    src/sst/elements/SnnDL/SnnPE.h \
    src/sst/elements/SnnDL/SpikeSource.h \
    src/sst/elements/SnnDL/SpikeEvent.h \
    src/sst/elements/SnnDL/SnnInterface.h \
    src/sst/elements/SnnDL/SnnNIC.h

# Documentation
doc_DATA = README.md LICENSE

# Examples and tests
EXTRA_DIST = \
    tests/test_minimal_snn.py \
    tests/test_network_spikes.py \
    tests/test_spikes.txt \
    examples/ \
    docs/

# Clean up
CLEANFILES = *.so *.la *.lo *.o

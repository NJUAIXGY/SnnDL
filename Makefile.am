# The v5 trace sources parse JSONL with the nlohmann single-header library that
# sst-core vendors under external/nlohmann.  sst-core distributes that header but
# does not install it, so it is absent from $(SST_PREFIX)/include and must be
# taken from the sibling sst-core checkout.  This is the only place that path is
# recorded; override it for non-standard layouts, e.g.
#   make SNNDL_SST_CORE_SRCDIR=/path/to/sst-core
SNNDL_SST_CORE_SRCDIR ?= $(top_srcdir)/../sst-core

AM_CPPFLAGS += \
	-I$(top_srcdir)/src \
	-I$(srcdir) \
	-I$(srcdir)/api \
	-I$(srcdir)/events \
	-I$(srcdir)/components \
	-I$(srcdir)/platform/memory \
	-I$(srcdir)/platform/noc \
	-I$(srcdir)/platform/core \
	-I$(srcdir)/components/timestep \
	-I$(srcdir)/snn/compute \
	-I$(srcdir)/snn/profiling \
	-I$(srcdir)/snn/synapse/route \
	-I$(srcdir)/research \
	-I$(SST_PREFIX)/include \
	-I$(SNNDL_SST_CORE_SRCDIR)/external

compdir = $(pkglibdir)
include build/core_sources.am
include build/extension_sources.am
include build/nextgen_sources.am
include build/v5_sources.am
include build/v5_core_sources.am

# These manifests are the reviewable active-build boundary.  The former PE and
# workload graph is retained under archive/legacy_gas/ only.
EXTRA_DIST = \
	build/core_sources.am \
	build/extension_sources.am \
	build/nextgen_sources.am \
	build/v5_sources.am \
	build/v5_core_sources.am \
	tools/check_build_boundaries.py \
	tools/run_nextgen_core_tests.sh \
	archive/legacy_gas/README.md \
	docs/SNNDL_V4_STORAGE_CONTRACT.md

comp_LTLIBRARIES = \
	libSnnDLCore.la \
	libSnnDLComm.la \
	libSnnDLLocal.la \
	libSnnDLRegistry.la \
	libSnnDLResearch.la \
	libSnnDLPlatform2D.la \
	libSnnDLNextCore.la \
	libSnnDLV5Contracts.la \
	libSnnDLV5Core.la \
	libSnnDL.la

libSnnDLCore_la_LDFLAGS = -module -avoid-version -L$(builddir)

libSnnDLComm_la_LDFLAGS = -module -avoid-version -L$(builddir)
libSnnDLComm_la_LIBADD = libSnnDLCore.la libSnnDLRegistry.la

libSnnDLLocal_la_LDFLAGS = -module -avoid-version -L$(builddir)

libSnnDLRegistry_la_LDFLAGS = -module -avoid-version -L$(builddir)

libSnnDLResearch_la_LDFLAGS = -module -avoid-version -L$(builddir)
libSnnDLResearch_la_LIBADD = libSnnDLCore.la libSnnDLComm.la libSnnDLRegistry.la

# SST-native schema-v4 2D platform.  This target owns the event graph and
# links only against the GAS-free domain core; legacy PE/workload sources are
# intentionally absent.
libSnnDLPlatform2D_la_SOURCES = \
	events/TimestepControlEvent.h \
	events/SnnMeshEvents.h \
	platform/core/SnnCoreTile.cc \
	platform/core/SnnCoreTile.h \
	components/MeshPE2D.cc \
	components/MeshPE2D.h \
	components/timestep/TimestepCoordinator.cc \
	components/timestep/TimestepCoordinator.h
libSnnDLPlatform2D_la_LDFLAGS = -module -avoid-version -L$(builddir)
libSnnDLPlatform2D_la_LIBADD = libSnnDLNextCore.la

# Pure synchronous timestep/domain contract.  It has no SST, workload, or
# research dependency and is tested independently below.
libSnnDLNextCore_la_SOURCES = $(SNNDL_NEXT_CORE_SOURCES)
libSnnDLNextCore_la_LDFLAGS = -module -avoid-version -L$(builddir)

# Header-only v5 contract types are compiled into a separate non-ELI library.
# It is intentionally not linked into the canonical v4 aggregate.
libSnnDLV5Contracts_la_SOURCES = $(SNNDL_V5_CONTRACT_SOURCES)
libSnnDLV5Contracts_la_LDFLAGS = -module -avoid-version -L$(builddir)

# P1 cycle pipeline.  The pure pipeline has no legacy dependency; the ELI
# component and ideal source are loaded through this independent library.
libSnnDLV5Core_la_SOURCES = $(SNNDL_V5_CORE_SOURCES)
libSnnDLV5Core_la_LDFLAGS = -module -avoid-version -L$(builddir)

# Aggregate compatibility name for SST installations.  The implementation is
# intentionally an empty anchor; all active ELI objects live in the libraries
# above, while no archived target is linked here.
libSnnDL_la_SOURCES = components/SnnDLCompatibility.cc
# SST loads the aggregate `libSnnDL.so` by element-library name.  Keep the
# domain libraries as DT_NEEDED entries so their ELI registration constructors
# are loaded with that entry point; otherwise the empty compatibility anchor
# hides all active components from Python configurations.
libSnnDL_la_LDFLAGS = -module -avoid-version -Wl,--no-as-needed -L$(builddir)
libSnnDL_la_LIBADD = \
	libSnnDLPlatform2D.la \
	libSnnDLV5Core.la

TEST_LINK = $(CXXLD) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(CXXFLAGS)

.PHONY: check-boundaries test-compile test-timestep-core test-v5-core-pipeline test-v5-cuba-lif \
	test-v5-core-storage test-v5-ca5a test-v5-ca5b test-v5-banked-sram test-v5-trace-jsonl-reader \
	test-bcsr-source-contract test-v5-address test-v5-statistics test-v5-multicast-tree \
	test-banked-sram test-local-storage

check-boundaries:
	python3 $(srcdir)/tools/check_build_boundaries.py --root $(srcdir)

test-timestep-core:
	$(srcdir)/tools/run_nextgen_core_tests.sh

test-bcsr-source-contract:
	$(TEST_LINK) $(srcdir)/tests/test_bcsr_source_contract.cc \
		$(srcdir)/platform/registry/BcsrSourceContract.cc -pthread \
		-o $(builddir)/test_bcsr_source_contract$(EXEEXT) && \
	$(builddir)/test_bcsr_source_contract$(EXEEXT)

test-v5-address:
	$(TEST_LINK) $(srcdir)/tests/test_v5_address.cc \
		$(srcdir)/v5/api/ArtifactContract.cc \
		-o $(builddir)/test_v5_address$(EXEEXT) && \
	$(builddir)/test_v5_address$(EXEEXT)

test-v5-statistics:
	$(TEST_LINK) $(srcdir)/tests/test_v5_statistics_contract.cc \
		-o $(builddir)/test_v5_statistics_contract$(EXEEXT) && \
	$(builddir)/test_v5_statistics_contract$(EXEEXT)

test-v5-multicast-tree:
	$(TEST_LINK) $(srcdir)/tests/v5/test_multicast_branch_table_v5.cc \
		-o $(builddir)/test_multicast_branch_table_v5$(EXEEXT) && \
	$(builddir)/test_multicast_branch_table_v5$(EXEEXT)

test-v5-core-pipeline:
	$(TEST_LINK) $(srcdir)/tests/v5/test_core_pipeline.cc \
	$(srcdir)/v5/core/CorePipeline.cc \
	$(srcdir)/v5/core/DeterministicRetireQueue.cc \
		$(srcdir)/v5/core/LifNeuronOp.cc \
		$(srcdir)/v5/core/IfNeuronOp.cc \
		$(srcdir)/v5/core/CubaLifNeuronOp.cc \
		$(srcdir)/v5/storage/CoreStorageV5.cc \
		$(srcdir)/v5/storage/StateDeltaLayoutV5.cc \
		$(srcdir)/v5/storage/BankedSramV5.cc \
		-o $(builddir)/test_core_pipeline$(EXEEXT) && \
	$(builddir)/test_core_pipeline$(EXEEXT)

test-v5-cuba-lif:
	$(TEST_LINK) $(srcdir)/tests/v5/test_cuba_lif_neuron_op.cc \
		$(srcdir)/v5/core/CubaLifNeuronOp.cc \
		-o $(builddir)/test_cuba_lif_neuron_op$(EXEEXT) && \
	$(builddir)/test_cuba_lif_neuron_op$(EXEEXT)

test-v5-if:
	$(TEST_LINK) $(srcdir)/tests/v5/test_if_neuron_op.cc \
		$(srcdir)/v5/core/IfNeuronOp.cc \
		$(srcdir)/v5/core/LifNeuronOp.cc \
		-o $(builddir)/test_if_neuron_op$(EXEEXT) && \
	$(builddir)/test_if_neuron_op$(EXEEXT)

test-v5-core-storage:
	$(TEST_LINK) $(srcdir)/tests/v5/test_core_storage_v5.cc \
		$(srcdir)/v5/storage/CoreStorageV5.cc \
		$(srcdir)/v5/storage/StateDeltaLayoutV5.cc \
		$(srcdir)/v5/core/CubaLifNeuronOp.cc \
		$(srcdir)/v5/storage/BankedSramV5.cc \
		-o $(builddir)/test_core_storage_v5$(EXEEXT) && \
	$(builddir)/test_core_storage_v5$(EXEEXT)

test-v5-ca5a:
	$(TEST_LINK) $(srcdir)/tests/v5/test_ca5a_sram_completion.cc \
		$(srcdir)/v5/core/CorePipeline.cc \
		$(srcdir)/v5/core/DeterministicRetireQueue.cc \
		$(srcdir)/v5/core/LifNeuronOp.cc \
		$(srcdir)/v5/core/IfNeuronOp.cc \
		$(srcdir)/v5/core/CubaLifNeuronOp.cc \
		$(srcdir)/v5/storage/CoreStorageV5.cc \
		$(srcdir)/v5/storage/StateDeltaLayoutV5.cc \
		$(srcdir)/v5/storage/BankedSramV5.cc \
		-o $(builddir)/test_ca5a_sram_completion$(EXEEXT) && \
	$(builddir)/test_ca5a_sram_completion$(EXEEXT)

test-v5-ca5b:
	PYTHONPATH=$(abs_top_srcdir)/../.. python3 -c 'from snndl.plans.state_delta_layout import write_ca5b_fixtures; write_ca5b_fixtures("/tmp")'
	$(TEST_LINK) $(srcdir)/tests/v5/test_ca5b_state_delta_layout.cc \
		$(srcdir)/v5/core/CorePipeline.cc \
		$(srcdir)/v5/core/DeterministicRetireQueue.cc \
		$(srcdir)/v5/core/LifNeuronOp.cc \
		$(srcdir)/v5/core/IfNeuronOp.cc \
		$(srcdir)/v5/core/CubaLifNeuronOp.cc \
		$(srcdir)/v5/storage/CoreStorageV5.cc \
		$(srcdir)/v5/storage/StateDeltaLayoutV5.cc \
		$(srcdir)/v5/storage/BankedSramV5.cc \
		-o $(builddir)/test_ca5b_state_delta_layout$(EXEEXT) && \
	$(builddir)/test_ca5b_state_delta_layout$(EXEEXT)

test-v5-banked-sram:
	$(TEST_LINK) $(srcdir)/tests/v5/test_banked_sram_v5.cc \
		$(srcdir)/v5/storage/BankedSramV5.cc \
		-o $(builddir)/test_banked_sram_v5$(EXEEXT) && \
	$(builddir)/test_banked_sram_v5$(EXEEXT)

# First direct unit test of the v5 trace runtime: the bounded JSONL record
# reader shared by the three replay sources.  It has no SST dependency, so it
# links standalone in the same style as the other test-v5-* targets.
test-v5-trace-jsonl-reader:
	$(TEST_LINK) $(srcdir)/tests/v5/test_trace_jsonl_reader_v5.cc \
		$(srcdir)/v5/trace/TraceJsonlReader.cc \
		-o $(builddir)/test_trace_jsonl_reader_v5$(EXEEXT) && \
	$(builddir)/test_trace_jsonl_reader_v5$(EXEEXT)

test-banked-sram:
	$(TEST_LINK) $(srcdir)/tests/test_banked_sram_model.cc \
		$(srcdir)/platform/memory/sram_sim/model/BankedSramModel.cc \
		-o $(builddir)/test_banked_sram_model$(EXEEXT) && \
	$(builddir)/test_banked_sram_model$(EXEEXT)

test-local-storage:
	$(TEST_LINK) $(srcdir)/tests/test_local_storage_hierarchy.cc \
		$(srcdir)/research/local_storage/LocalStorageHierarchyController.cc \
		$(srcdir)/research/local_storage/PeWeightObjectPlane.cc \
		$(srcdir)/platform/memory/sram_sim/model/BankedSramModel.cc \
		-o $(builddir)/test_local_storage_hierarchy$(EXEEXT) && \
	$(builddir)/test_local_storage_hierarchy$(EXEEXT)

test-compile:
	$(CXXCOMPILE) -c $(srcdir)/tests/test_bcsr_source_contract.cc -o $(builddir)/test_bcsr_source_contract.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_banked_sram_model.cc -o $(builddir)/test_banked_sram_model.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_local_storage_hierarchy.cc -o $(builddir)/test_local_storage_hierarchy.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_v5_address.cc -o $(builddir)/test_v5_address.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_v5_statistics_contract.cc -o $(builddir)/test_v5_statistics_contract.o
	$(CXXCOMPILE) -c $(srcdir)/tests/v5/test_multicast_branch_table_v5.cc -o $(builddir)/test_multicast_branch_table_v5.o
	$(CXXCOMPILE) -c $(srcdir)/tests/v5/test_core_pipeline.cc -o $(builddir)/test_core_pipeline.o
	$(CXXCOMPILE) -c $(srcdir)/tests/v5/test_core_storage_v5.cc -o $(builddir)/test_core_storage_v5.o
	$(CXXCOMPILE) -c $(srcdir)/tests/v5/test_trace_jsonl_reader_v5.cc -o $(builddir)/test_trace_jsonl_reader_v5.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_timestep_core.cc -o $(builddir)/test_timestep_core.o

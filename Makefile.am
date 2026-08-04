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
	-I$(SST_PREFIX)/include

compdir = $(pkglibdir)
include build/core_sources.am
include build/extension_sources.am
include build/nextgen_sources.am

# These manifests are the reviewable active-build boundary.  The former PE and
# workload graph is retained under archive/legacy_gas/ only.
EXTRA_DIST = \
	build/core_sources.am \
	build/extension_sources.am \
	build/nextgen_sources.am \
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
	libSnnDLCore.la \
	libSnnDLComm.la \
	libSnnDLLocal.la \
	libSnnDLRegistry.la \
	libSnnDLPlatform2D.la

TEST_LINK = $(CXXLD) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(CXXFLAGS)

.PHONY: check-boundaries test-compile test-timestep-core \
	test-bcsr-source-contract test-dma-scheduler test-banked-sram \
	test-local-storage

check-boundaries:
	python3 $(srcdir)/tools/check_build_boundaries.py --root $(srcdir)

test-timestep-core:
	$(srcdir)/tools/run_nextgen_core_tests.sh

test-bcsr-source-contract:
	$(TEST_LINK) $(srcdir)/tests/test_bcsr_source_contract.cc \
		$(srcdir)/platform/registry/BcsrSourceContract.cc -pthread \
		-o $(builddir)/test_bcsr_source_contract$(EXEEXT) && \
	$(builddir)/test_bcsr_source_contract$(EXEEXT)

test-dma-scheduler:
	$(TEST_LINK) $(srcdir)/tests/test_pe_dma_scheduler.cc \
		$(srcdir)/platform/memory/DmaMemAccessProxy.cc \
		$(srcdir)/platform/memory/PeDmaScheduler.cc \
		-o $(builddir)/test_pe_dma_scheduler$(EXEEXT) && \
	$(builddir)/test_pe_dma_scheduler$(EXEEXT)

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
	$(CXXCOMPILE) -c $(srcdir)/tests/test_pe_dma_scheduler.cc -o $(builddir)/test_pe_dma_scheduler.o
	$(CXXCOMPILE) -c $(srcdir)/tests/test_timestep_core.cc -o $(builddir)/test_timestep_core.o

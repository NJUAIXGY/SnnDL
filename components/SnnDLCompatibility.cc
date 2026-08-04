// Compatibility translation unit for the aggregate libSnnDL plugin name.
// Keep one symbol from every active domain reachable so libtool emits the
// domain libraries as DT_NEEDED entries and SST loads their ELI registrars.
extern "C" void snndl_core_anchor();
extern "C" void snndl_comm_anchor();
extern "C" void snndl_local_anchor();
extern "C" void snndl_registry_anchor();
extern "C" void snndl_research_anchor();
extern "C" void snndl_platform2d_anchor();

extern "C" void snndl_compatibility_anchor() {
    snndl_core_anchor();
    snndl_comm_anchor();
    snndl_local_anchor();
    snndl_registry_anchor();
    snndl_research_anchor();
    snndl_platform2d_anchor();
}

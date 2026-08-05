// Compatibility translation unit for the canonical aggregate libSnnDL plugin
// name.  The frozen v4 2D platform and the independent v5 core are reachable
// from this entry point; other extensions remain explicit libraries.
extern "C" void snndl_platform2d_anchor();
extern "C" void snndl_v5_core_anchor();

extern "C" void snndl_compatibility_anchor() {
    snndl_platform2d_anchor();
    snndl_v5_core_anchor();
}

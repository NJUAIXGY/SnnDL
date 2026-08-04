// Keeps the compatibility plugin linked to the optional implementation.
// Without a real symbol reference, --as-needed may discard libSnnDLOpt.
extern "C" void snndl_extension_anchor() {}

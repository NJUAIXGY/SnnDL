# Compute Domain

`SnnCoreEngine` and `SnnNeuronModel` implement the active neuron-state update
and firing policy.  `SnnLearningCore` is an optional learning helper with a
callback-based writeback boundary; it does not own memory or network objects.
`SynapseManager` and `SnnProfiler` are lightweight data/profiling utilities.

The next-generation synchronous closure is intentionally independent:
`snn/timestep/TimestepTracker`, `DeltaAccumulator`, and
`NextGenNeuronEngine` are built in `libSnnDLNextCore.la` and use only
`api/TimestepTypes.h` plus the C++ standard library.

Keep compute APIs free of SST component assembly and transport policy.  Validate
changes with `make test-timestep-core` and the active compile target.

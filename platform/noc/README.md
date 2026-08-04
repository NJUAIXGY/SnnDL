# Active NoC Platform

`NocSubsystem` owns packet queues, endpoint delivery callbacks, and the
internal ring scheduler.  `OptimizedInternalRing` supplies the deterministic
2D ring backend.  Both operate on `NocPacketEvent` and do not decode BCSR or
neuron semantics.

SST components in `components/noc/` adapt these callbacks to directional links,
SimpleNetwork, or the explicit 2D multicast experiment backend.  Route and
stimulus domains use `api/INocTransport` rather than reaching into this
implementation.

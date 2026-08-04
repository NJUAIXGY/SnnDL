# Active Events

The event layer contains serializable transport and timestep data only.  It
does not own neuron state, routing policy, or memory scheduling.

- `NocPacketEvent.h` and `NocPacketBatchEvent.h`: payload-agnostic 2D packet
  carriers.
- `SpikeEvent.h`, `SpikePacket.h`, and `SpikeEventWrapper.{h,cc}`: spike
  payloads and SimpleNetwork adapters.
- `TimestepControlEvent.h`: timestep start/retire control data for the
  next-generation synchronous core.
- `SimpleTestEvent.{h,cc}`: minimal link and serialization smoke event.

Keep event fields stable and serializable.  If a value is only needed by a
compute or route implementation, keep it in that domain rather than extending
the packet ABI.

Historical control-plane events are retained under `archive/legacy_gas/events/`.

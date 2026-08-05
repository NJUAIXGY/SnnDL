#ifndef SST_SNN_DL_V5_NOC_PROBE_V5_H
#define SST_SNN_DL_V5_NOC_PROBE_V5_H
#include "NocEventsV5.h"
#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <cstdint>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {
class NocProbeV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(NocProbeV5,"SnnDL","NocProbeV5",SST_ELI_ELEMENT_VERSION(1,0,0),"P4 deterministic Merlin traffic probe",COMPONENT_CATEGORY_NETWORK)
    SST_ELI_DOCUMENT_PARAMS(
        {"pe_id","Source PE","0"},{"destination_pe","Destination PE","0"},
        {"packets","Packets to inject","0"},{"expected_packets","Packets expected at this probe","0"},
        {"payload_bytes","Payload bytes beyond fixed v5 header","0"},{"start_cycle","First injection cycle","1"},
        {"injection_gap_cycles","Gap between accepted injections","0"},{"timestep","Packet timestep","0"},
        {"output_json","Probe evidence path",""},{"clock","Probe clock","1GHz"},{"verbose","Verbose level","0"})
    SST_ELI_DOCUMENT_PORTS(
        {"inject","Packet output to endpoint",{"SnnDL.NocPacketV5Event"}},
        {"receive","Injection ACK and delivered packets",{"SnnDL.NocInjectionAckV5Event","SnnDL.NocPacketV5Event"}})
    NocProbeV5(SST::ComponentId_t,SST::Params&); ~NocProbeV5() override;
    void finish() override;
private:
    void handle_(SST::Event*); bool tick_(SST::Cycle_t); void send_(); void checkDone_(); void write_() const;
    SST::Output out_; SST::Link *inject_=nullptr,*receive_=nullptr; std::uint32_t pe_=0,dest_=0,payload_=0;
    std::uint64_t packets_=0,expected_=0,start_=1,gap_=0,timestep_=0,cycle_=0,sent_=0,received_=0,retries_=0,next_cycle_=0;
    bool inflight_=false,done_=false; std::uint64_t inflight_id_=0,latency_sum_=0,latency_max_=0; std::string output_;
};
}}}
#endif

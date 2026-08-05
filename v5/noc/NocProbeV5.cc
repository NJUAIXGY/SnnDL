#include <sst/core/sst_config.h>
#include "NocProbeV5.h"
#include <algorithm>
#include <fstream>
namespace SST { namespace SnnDL { namespace v5 {
NocProbeV5::NocProbeV5(SST::ComponentId_t id,SST::Params& p):Component(id),out_("SnnDL.NocProbeV5",0,0,Output::STDOUT),
 pe_(p.find<std::uint32_t>("pe_id",0)),dest_(p.find<std::uint32_t>("destination_pe",0)),payload_(p.find<std::uint32_t>("payload_bytes",0)),
 packets_(p.find<std::uint64_t>("packets",0)),expected_(p.find<std::uint64_t>("expected_packets",0)),start_(p.find<std::uint64_t>("start_cycle",1)),
 gap_(p.find<std::uint64_t>("injection_gap_cycles",0)),timestep_(p.find<std::uint64_t>("timestep",0)),output_(p.find<std::string>("output_json","")){
 out_.setVerboseLevel(p.find<int>("verbose",0));inject_=configureLink("inject");receive_=configureLink("receive",new Event::Handler2<NocProbeV5,&NocProbeV5::handle_>(this));
 if(!inject_||!receive_)out_.fatal(CALL_INFO,-1,"NocProbeV5 requires inject and receive links\n");
 registerClock(p.find<std::string>("clock","1GHz"),new Clock::Handler2<NocProbeV5,&NocProbeV5::tick_>(this));registerAsPrimaryComponent();primaryComponentDoNotEndSim();
}
NocProbeV5::~NocProbeV5()=default;
void NocProbeV5::send_(){auto* e=new NocPacketV5Event();e->packet_id=(std::uint64_t(pe_)<<48)|sent_;e->timestep=timestep_;e->source_pe=pe_;e->source_neuron=std::uint32_t(sent_);e->destination_pe=dest_;e->source_event_seq=sent_+1;e->payload_bytes=payload_;inflight_id_=e->packet_id;inflight_=true;inject_->send(e);}
void NocProbeV5::handle_(SST::Event* e){
 if(auto* a=dynamic_cast<NocInjectionAckV5Event*>(e)){if(!inflight_||a->packet_id!=inflight_id_){delete a;out_.fatal(CALL_INFO,-1,"probe ACK mismatch\n");}inflight_=false;if(a->accepted){++sent_;next_cycle_=cycle_+gap_+1;}else if(a->retryable)++retries_;else{delete a;out_.fatal(CALL_INFO,-1,"probe injection permanently rejected\n");}delete a;}
 else if(auto* p=dynamic_cast<NocPacketV5Event*>(e)){const auto now=std::uint64_t(getCurrentSimTimeNano());const auto lat=now>=p->injection_time_ns?now-p->injection_time_ns:0;latency_sum_+=lat;latency_max_=std::max(latency_max_,lat);++received_;delete p;}
 else{delete e;out_.fatal(CALL_INFO,-1,"probe received unexpected event\n");}checkDone_();
}
void NocProbeV5::checkDone_(){if(!done_&&sent_==packets_&&!inflight_&&received_==expected_){done_=true;primaryComponentOKToEndSim();}}
bool NocProbeV5::tick_(SST::Cycle_t){++cycle_;if(!done_&&!inflight_&&sent_<packets_&&cycle_>=start_&&cycle_>=next_cycle_)send_();checkDone_();return done_;}
void NocProbeV5::write_()const{if(output_.empty())return;std::ofstream f(output_);f<<"{\n  \"pe_id\": "<<pe_<<",\n  \"destination_pe\": "<<dest_<<",\n  \"payload_bytes\": "<<payload_<<",\n  \"packets_sent\": "<<sent_<<",\n  \"packets_received\": "<<received_<<",\n  \"expected_packets\": "<<expected_<<",\n  \"injection_retries\": "<<retries_<<",\n  \"latency_sum_ns\": "<<latency_sum_<<",\n  \"latency_max_ns\": "<<latency_max_<<",\n  \"completion_cycle\": "<<cycle_<<"\n}\n";}
void NocProbeV5::finish(){write_();}
}}}

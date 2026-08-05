#include <sst/core/sst_config.h>
#include "PeEndpointV5.h"
#include <algorithm>
#include <fstream>

namespace SST { namespace SnnDL { namespace v5 {

PeEndpointV5::PeEndpointV5(SST::ComponentId_t id, SST::Params& p)
    : Component(id), out_("SnnDL.PeEndpointV5",0,0,Output::STDOUT),
      pe_id_(p.find<std::uint32_t>("pe_id",0)), mesh_x_(std::max(1u,p.find<std::uint32_t>("mesh_x",1))),
      tx_capacity_(std::max(1u,p.find<std::uint32_t>("tx_queue_entries",16))),
      rx_capacity_(std::max(1u,p.find<std::uint32_t>("rx_queue_entries",16))),
      flit_bytes_(std::max(1u,p.find<std::uint32_t>("flit_size_bytes",32))),
      destination_pe_(p.find<std::uint32_t>("destination_pe",0)), destination_core_(p.find<std::uint32_t>("destination_core",0)),
      payload_bytes_(p.find<std::uint32_t>("payload_bytes",0)), core_attached_(p.find<int>("core_attached",0)!=0),
      core_held_capacity_(std::max(1u,p.find<std::uint32_t>("core_held_spike_entries",32))),
      output_json_(p.find<std::string>("output_json","")) {
    out_.setVerboseLevel(p.find<int>("verbose",0));
    network_ = loadUserSubComponent<SST::Interfaces::SimpleNetwork>("networkIF", ComponentInfo::SHARE_NONE, 1);
    if (!network_) out_.fatal(CALL_INFO,-1,"PeEndpointV5 requires networkIF=merlin.linkcontrol\n");
    probe_in_=configureLink("probe_in",new Event::Handler2<PeEndpointV5,&PeEndpointV5::handleProbe_>(this));
    probe_out_=configureLink("probe_out");
    provider_control_=configureLink("provider_control",new Event::Handler2<PeEndpointV5,&PeEndpointV5::handleProviderControl_>(this));
    core_control_=configureLink("core_control");
    provider_spike_=configureLink("provider_spike",new Event::Handler2<PeEndpointV5,&PeEndpointV5::handleProviderSpike_>(this));
    provider_ack_=configureLink("provider_ack"); core_spike_=configureLink("core_spike");
    core_ack_=configureLink("core_ack",new Event::Handler2<PeEndpointV5,&PeEndpointV5::handleCoreAck_>(this));
    core_egress_=configureLink("core_egress",new Event::Handler2<PeEndpointV5,&PeEndpointV5::handleCoreEgress_>(this));
    provider_monitor_=configureLink("provider_monitor");
    if (core_attached_ && (!provider_control_||!core_control_||!core_spike_||!core_ack_))
        out_.fatal(CALL_INFO,-1,"core-attached PeEndpointV5 is missing proxy links\n");
    if (core_attached_ && tx_capacity_ < core_held_capacity_)
        out_.fatal(CALL_INFO,-1,"core-attached PeEndpointV5 TX capacity must cover one held-spike release burst\n");
    registerClock(p.find<std::string>("clock","1GHz"),new Clock::Handler2<PeEndpointV5,&PeEndpointV5::tick_>(this));
    registerStatistic<std::uint64_t>("noc.packets"); registerStatistic<std::uint64_t>("noc.flits");
    registerStatistic<std::uint64_t>("noc.logical_deliveries"); registerStatistic<std::uint64_t>("noc.tx_stall_cycles");
    registerStatistic<std::uint64_t>("noc.rx_stall_cycles"); registerStatistic<std::uint64_t>("noc.core_retry_cycles");
}
PeEndpointV5::~PeEndpointV5(){ for(auto* p:tx_)delete p; for(auto* p:rx_)delete p; delete pending_seal_; delete network_; }
void PeEndpointV5::init(unsigned int phase){ network_->init(phase); }
void PeEndpointV5::setup(){ network_->setup(); if(network_->getEndpointID()!=pe_id_) out_.fatal(CALL_INFO,-1,"endpoint/linkcontrol nid mismatch\n"); }

void PeEndpointV5::enqueue_(NocPacketV5Event* packet, SST::Link* ack_link){
    auto* ack=new NocInjectionAckV5Event(); ack->packet_id=packet->packet_id;
    if(packet->format_version!=kNocPacketV5FormatVersion){ ack->accepted=false; ack->retryable=false; delete packet; }
    else if(tx_.size()>=tx_capacity_){ ack->accepted=false; ack->retryable=true; delete packet; }
    else { packet->source_pe=pe_id_; packet->injection_time_ns=getCurrentSimTimeNano(); tx_.push_back(packet); ack->accepted=true; }
    if(ack_link) ack_link->send(ack); else delete ack;
}
void PeEndpointV5::handleProbe_(SST::Event* e){ auto* p=dynamic_cast<NocPacketV5Event*>(e); if(!p){delete e;out_.fatal(CALL_INFO,-1,"bad probe event\n");} enqueue_(p,probe_out_); }
void PeEndpointV5::handleProviderSpike_(SST::Event* e){
    auto* s=dynamic_cast<CoreSpikeEvent*>(e); if(!s){delete e;out_.fatal(CALL_INFO,-1,"bad provider spike\n");}
    if(!core_spike_){delete s;return;} core_spike_->send(s);
}
void PeEndpointV5::handleProviderControl_(SST::Event* e){
    auto* c=dynamic_cast<CoreControlEvent*>(e); if(!c){delete e;out_.fatal(CALL_INFO,-1,"bad control\n");}
    if(c->operation==CoreControlOp::Start){ core_started_=true; core_control_->send(c); }
    else if(c->operation==CoreControlOp::SealIngress && (!rx_.empty()||core_inflight_)){ delete pending_seal_; pending_seal_=c; }
    else core_control_->send(c);
}
void PeEndpointV5::handleCoreAck_(SST::Event* e){
    auto* a=dynamic_cast<CoreSpikeAckEvent*>(e); if(!a){delete e;out_.fatal(CALL_INFO,-1,"bad core ack\n");}
    if(core_inflight_){
        if(a->accepted){ delete rx_.front(); rx_.pop_front(); ++logical_deliveries_; core_inflight_=false; }
        else if(a->retryable){ ++core_retries_; core_inflight_=false; }
        else { delete a; out_.fatal(CALL_INFO,-1,"Core permanently rejected network packet\n"); }
        delete a;
    } else if(provider_ack_) provider_ack_->send(a); else delete a;
}
void PeEndpointV5::handleCoreEgress_(SST::Event* e){
    auto* s=dynamic_cast<CoreSpikeEvent*>(e); if(!s){delete e;out_.fatal(CALL_INFO,-1,"bad core egress\n");}
    if(provider_monitor_) provider_monitor_->send(s->clone());
    auto* p=new NocPacketV5Event(); p->packet_id=(std::uint64_t(pe_id_)<<48)^s->source_event_seq;
    p->timestep=s->timestep;p->source_pe=pe_id_;p->source_neuron=s->source_neuron;p->source_event_seq=s->source_event_seq;
    p->destination_pe=destination_pe_;p->destination_core=destination_core_;p->target_neuron=s->target_neuron;p->payload_bytes=payload_bytes_;
    if(tx_.size()>=tx_capacity_){ delete p; delete s; out_.fatal(CALL_INFO,-1,"Core egress exceeds finite endpoint TX capacity\n"); }
    p->injection_time_ns=getCurrentSimTimeNano();tx_.push_back(p);delete s;
}
void PeEndpointV5::transmit_(){
    if(tx_.empty())return; auto* p=tx_.front();
    if(p->destination_pe==pe_id_){ tx_.pop_front(); if(rx_.size()<rx_capacity_)rx_.push_back(p);else out_.fatal(CALL_INFO,-1,"local loopback RX overflow\n"); return; }
    const auto bits=p->wireBytes()*8; if(!network_->spaceToSend(0,bits)){++tx_stalls_;return;}
    auto* req=new SST::Interfaces::SimpleNetwork::Request(p->destination_pe,pe_id_,bits,true,true,p);
    req->vn=0;req->allow_adaptive=false;network_->send(req,0);tx_.pop_front();++tx_packets_;tx_bits_+=bits;tx_flits_+=(p->wireBytes()+flit_bytes_-1)/flit_bytes_;
}
void PeEndpointV5::receive_(){
    if(rx_.size()>=rx_capacity_){if(network_->requestToReceive(0))++rx_stalls_;return;}
    if(!network_->requestToReceive(0))return; auto* req=network_->recv(0); if(!req)return;
    auto* p=dynamic_cast<NocPacketV5Event*>(req->takePayload()); if(!p||p->format_version!=kNocPacketV5FormatVersion||p->destination_pe!=pe_id_){delete p;delete req;out_.fatal(CALL_INFO,-1,"invalid received v5 packet\n");}
    const auto now=std::uint64_t(getCurrentSimTimeNano()); const auto latency=now>=p->injection_time_ns?now-p->injection_time_ns:0;
    const auto sx=p->source_pe%mesh_x_,sy=p->source_pe/mesh_x_,dx=pe_id_%mesh_x_,dy=pe_id_/mesh_x_;
    const auto hops=std::uint64_t(sx>dx?sx-dx:dx-sx)+std::uint64_t(sy>dy?sy-dy:dy-sy);
    latency_sum_ns_+=latency;latency_max_ns_=std::max(latency_max_ns_,latency);hop_sum_+=hops;hop_max_=std::max(hop_max_,hops);++rx_packets_;rx_.push_back(p);delete req;
}
void PeEndpointV5::dispatch_(){
    if(rx_.empty())return; auto* p=rx_.front();
    if(core_attached_){ if(!core_started_||core_inflight_)return; auto* s=new CoreSpikeEvent();s->timestep=p->timestep;s->source_neuron=p->source_neuron;s->target_neuron=p->target_neuron;s->source_event_seq=p->source_event_seq;core_spike_->send(s);core_inflight_=true; }
    else { if(probe_out_)probe_out_->send(p->clone());delete p;rx_.pop_front();++logical_deliveries_; }
}
bool PeEndpointV5::tick_(SST::Cycle_t){++cycles_;receive_();dispatch_();transmit_();if(pending_seal_&&rx_.empty()&&!core_inflight_){core_control_->send(pending_seal_);pending_seal_=nullptr;}return false;}
void PeEndpointV5::writeEvidence_() const { if(output_json_.empty())return;std::ofstream f(output_json_);f<<"{\n  \"pe_id\": "<<pe_id_<<",\n  \"cycles\": "<<cycles_<<",\n  \"tx_packets\": "<<tx_packets_<<",\n  \"rx_packets\": "<<rx_packets_<<",\n  \"logical_deliveries\": "<<logical_deliveries_<<",\n  \"tx_bits\": "<<tx_bits_<<",\n  \"tx_flits\": "<<tx_flits_<<",\n  \"tx_stall_cycles\": "<<tx_stalls_<<",\n  \"rx_stall_cycles\": "<<rx_stalls_<<",\n  \"core_retry_cycles\": "<<core_retries_<<",\n  \"latency_sum_ns\": "<<latency_sum_ns_<<",\n  \"latency_max_ns\": "<<latency_max_ns_<<",\n  \"hop_sum\": "<<hop_sum_<<",\n  \"hop_max\": "<<hop_max_<<",\n  \"tx_queue_remaining\": "<<tx_.size()<<",\n  \"rx_queue_remaining\": "<<rx_.size()<<",\n  \"drops\": "<<drops_<<"\n}\n"; }
void PeEndpointV5::finish(){writeEvidence_();registerStatistic<std::uint64_t>("noc.packets")->addData(tx_packets_);registerStatistic<std::uint64_t>("noc.flits")->addData(tx_flits_);registerStatistic<std::uint64_t>("noc.logical_deliveries")->addData(logical_deliveries_);registerStatistic<std::uint64_t>("noc.tx_stall_cycles")->addData(tx_stalls_);registerStatistic<std::uint64_t>("noc.rx_stall_cycles")->addData(rx_stalls_);registerStatistic<std::uint64_t>("noc.core_retry_cycles")->addData(core_retries_);network_->finish();}

}}}

// -*- c++ -*-
//
// MulticastNIC implementation
//

#include "MulticastNIC.h"

#include <sst/core/event.h>
#include <sst/core/params.h>

namespace SST { namespace SnnDL {

MulticastNIC::MulticastNIC(SST::ComponentId_t id, SST::Params& params)
    : SnnInterface(id, params),
      out_("SnnDL.MulticastNIC", params.find<int>("verbose", 0), 0, SST::Output::STDOUT) {

    node_id_ = params.find<uint32_t>("node_id", 0);
    port_name_ = params.find<std::string>("port_name", "network");
    network_link_ = configureLink(port_name_,
                                  new SST::Event::Handler2<MulticastNIC, &MulticastNIC::handleNetworkLinkEvent_>(this));
}

MulticastNIC::~MulticastNIC() = default;

void MulticastNIC::init(unsigned int /*phase*/) {}
void MulticastNIC::setup() {}
void MulticastNIC::finish() {}

void MulticastNIC::setReceiveHandler(ReceiveHandler handler) {
    recv_handler_ = std::move(handler);
}

void MulticastNIC::sendToNode(uint32_t /*dest_node*/, SST::Event* event) {
    if (!event) return;
    if (!network_link_) {
        delete event;
        return;
    }
    network_link_->send(event);
}

void MulticastNIC::setNodeId(uint32_t node_id) {
    node_id_ = node_id;
}

uint32_t MulticastNIC::getNodeId() const {
    return node_id_;
}

std::string MulticastNIC::getNetworkStatus() const {
    return std::string("MulticastNIC(node_id=") + std::to_string(node_id_) + ", port=" + port_name_ + ")";
}

void MulticastNIC::handleNetworkLinkEvent_(SST::Event* ev) {
    if (!ev) return;
    if (!recv_handler_) {
        delete ev;
        return;
    }
    recv_handler_(ev); // handler takes ownership
}

}} // namespace SST::SnnDL


#ifndef SNNDL_SPIKE_RECEIVER_H
#define SNNDL_SPIKE_RECEIVER_H

#include <array>
#include <map>
#include <memory>
#include <cstdint>
#include <vector>
#include <sst/core/interfaces/simpleNetwork.h>

#include "PacketDecoder.h"
#include "ReceiveBuffer.h"

namespace SST { namespace SnnDL {

class SpikeReceiver {
public:
    explicit SpikeReceiver(SST::Interfaces::SimpleNetwork* net = nullptr,
                           uint32_t mesh_x = 0, uint32_t mesh_y = 0)
        : decoder_(std::make_unique<PacketDecoder>())
        , buffer_(std::make_unique<ReceiveBuffer>())
        , network_(net)
        , mesh_x_(mesh_x)
        , mesh_y_(mesh_y) {}

    bool receiveFromRouter(int virtual_channel) {
        (void)virtual_channel;
        if (!network_) return false;
        auto* req = network_->recv(0);
        if (!req) return true;
        handleNetworkRequest(req);
        delete req;
        return true;
    }

    void handleNetworkRequest(SST::Interfaces::SimpleNetwork::Request* req) {
        if (!req) return;
        // In a full implementation, extract payload and decode to spikes
        // Here we attempt to inspect payload and reconstruct a single spike event
        auto* payload = req->inspectPayload();
        if (!payload) return;
        // Minimal path: assume payload is already a SpikeEvent
        SpikeEvent* ev = dynamic_cast<SpikeEvent*>(payload);
        if (ev) {
            buffer_->enqueueSpike(ev);
        }
    }

    SpikeEvent* getNextSpike() { return buffer_ ? buffer_->dequeueSpike() : nullptr; }
    bool hasPendingSpikes() const { return buffer_ && buffer_->getSize() > 0; }

    bool isLocalDestination(uint32_t /*dest_neuron*/) const { return true; }
    uint32_t getDestinationPE(uint32_t /*dest_neuron*/) const { return 0; }

    void initMeshRouting(uint32_t /*pe_id*/) {
        // Stub: routing_table_ remains empty for now
    }

    void setNetwork(SST::Interfaces::SimpleNetwork* net) { network_ = net; }

private:
    std::unique_ptr<PacketDecoder> decoder_;
    std::unique_ptr<ReceiveBuffer> buffer_;
    SST::Interfaces::SimpleNetwork* network_;

    uint32_t mesh_x_ = 0, mesh_y_ = 0;
    std::array<uint32_t, 4> neighbor_ids_{};
    std::map<uint32_t, uint32_t> routing_table_;
};

}} // namespace

#endif // SNNDL_SPIKE_RECEIVER_H

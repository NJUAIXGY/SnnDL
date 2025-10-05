#ifndef SNNDL_PACKET_DECODER_H
#define SNNDL_PACKET_DECODER_H

#include <cstdint>
#include <vector>
#include <string>
#include "SpikePacket.h"

namespace SST { namespace SnnDL {

class PacketDecoder {
public:
    PacketDecoder() = default;

    // Decode raw bytes into spike events. For now, return empty list.
    std::vector<SpikeEvent*> decodePacket(const uint8_t* packet_data, size_t size) {
        (void)packet_data; (void)size; // unused
        return {};
    }

    bool validatePacket(const uint8_t* packet_data, size_t size) {
        (void)packet_data; (void)size;
        return true; // stub always valid
    }

    SpikePacketHeader extractHeader(const uint8_t* packet_data) {
        (void)packet_data;
        SpikePacketHeader h{};
        h.version = 1;
        return h;
    }
};

}} // namespace

#endif // SNNDL_PACKET_DECODER_H

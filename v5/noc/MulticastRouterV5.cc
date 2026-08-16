#include <sst/core/sst_config.h>

#include "MulticastRouterV5.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

std::uint8_t MulticastRouterV5::branchBit_(Port port) {
    switch (port) {
    case East: return 1u << 0;
    case West: return 1u << 1;
    case South: return 1u << 2;
    case North: return 1u << 3;
    case Local: return 1u << 4;
    default: return 0;
    }
}

MulticastRouterV5::MulticastRouterV5(SST::ComponentId_t id, SST::Params& p)
    : Component(id), out_("SnnDL.MulticastRouterV5", 0, 0, Output::STDOUT),
      pe_id_(p.find<std::uint32_t>("pe_id", 0)),
      rows_(std::max(1u, p.find<std::uint32_t>("mesh_rows", 1))),
      cols_(std::max(1u, p.find<std::uint32_t>("mesh_cols", 1))),
      input_entries_(std::max(1u, p.find<std::uint32_t>("input_queue_entries", 8))),
      flit_bytes_(std::max(1u, p.find<std::uint32_t>("flit_size_bytes", 32))),
      route_latency_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("route_latency_cycles", 1))),
      output_latency_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("output_latency_cycles", 1))),
      output_json_(p.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(p.find<int>("verbose", 0));
    if (pe_id_ >= rows_ * cols_) out_.fatal(CALL_INFO, -1, "MulticastRouterV5 PE is outside mesh\n");
    try { branches_ = parseMulticastBranchTableV5(p.find<std::string>("branch_table", "")); }
    catch (const std::exception& error) { out_.fatal(CALL_INFO, -1, "branch table error: %s\n", error.what()); }
    const char* names[PortCount] = {"local", "east", "west", "south", "north"};
    for (int port=0; port<PortCount; ++port) {
        links_[port] = configureLink(names[port], new Event::Handler2<MulticastRouterV5,
                                     &MulticastRouterV5::handle_, int>(this, port));
        credits_[port].reset(links_[port] ? input_entries_ : 0);
    }
    if (!links_[Local]) out_.fatal(CALL_INFO, -1, "MulticastRouterV5 requires a local endpoint link\n");
    for (const char* name : {"mcast.source_packets", "mcast.router_clones", "mcast.branch_transmissions",
                             "mcast.link_traversals", "mcast.flit_traversals", "mcast.route_lookups",
                             "mcast.credit_stall_cycles", "mcast.output_stall_cycles"})
        registerStatistic<std::uint64_t>(name);
    registerClock(p.find<std::string>("clock", "1GHz"),
                  new Clock::Handler2<MulticastRouterV5, &MulticastRouterV5::tick_>(this));
}

MulticastRouterV5::~MulticastRouterV5() {
    for (auto& queue : inputs_) for (auto& item : queue) delete item.packet;
}

void MulticastRouterV5::handle_(SST::Event* event, int port_value) {
    const auto port = static_cast<Port>(port_value);
    if (auto* credit = dynamic_cast<NocCreditV5Event*>(event)) {
        if (!credits_[port].restore(credit->credits)) {
            delete credit; out_.fatal(CALL_INFO, -1, "multicast credit overflow on port %d\n", port_value);
        }
        delete credit;
        return;
    }
    auto* packet = dynamic_cast<NocPacketV5Event*>(event);
    if (!packet || packet->format_version != kNocPacketV5FormatVersion) {
        delete event; out_.fatal(CALL_INFO, -1, "invalid multicast data event\n");
    }
    if (inputs_[port].size() >= input_entries_) {
        delete packet; ++drops_; out_.fatal(CALL_INFO, -1, "multicast input buffer overflow\n");
    }
    inputs_[port].push_back(QueuedPacket{packet, cycle_ + route_latency_});
    input_peak_ = std::max<std::uint64_t>(input_peak_, inputs_[port].size());
    if (port == Local) ++source_packets_;
}

bool MulticastRouterV5::tick_(SST::Cycle_t) {
    ++cycle_;
    for (std::size_t step=0; step<PortCount; ++step) {
        const auto input = static_cast<Port>((round_robin_ + step) % PortCount);
        if (inputs_[input].empty() || inputs_[input].front().ready_cycle > cycle_) continue;
        auto* packet = inputs_[input].front().packet;
        const auto found = branches_.find(packet->route_id);
        if (found == branches_.end()) out_.fatal(CALL_INFO, -1, "missing branch action for route=%llu pe=%u\n",
                                                  static_cast<unsigned long long>(packet->route_id), pe_id_);
        ++route_lookups_;
        std::vector<Port> outputs;
        for (Port port : {Local, East, West, South, North})
            if (found->second.output_mask & branchBit_(port)) outputs.push_back(port);
        if (outputs.empty()) {
            delete packet;
            inputs_[input].pop_front();
            links_[input]->send(new NocCreditV5Event());
            round_robin_ = (input + 1) % PortCount;
            continue;
        }
        bool blocked_credit=false, blocked_output=false;
        for (const auto output : outputs) {
            if (!links_[output] || credits_[output].available() == 0) blocked_credit=true;
            if (output_busy_until_[output] > cycle_) blocked_output=true;
        }
        if (blocked_credit || blocked_output) {
            if (blocked_credit) ++credit_stalls_;
            if (blocked_output) ++output_stalls_;
            continue;
        }
        const auto flits = (packet->wireBytes() + flit_bytes_ - 1) / flit_bytes_;
        for (const auto output : outputs) {
            auto* copy = packet->clone();
            if (output == Local) {
                copy->destination_pe = pe_id_;
                copy->destination_core_mask = found->second.local_core_mask;
            }
            links_[output]->send(copy);
            if (!credits_[output].consume())
                out_.fatal(CALL_INFO, -1, "multicast output credit underflow\n");
            output_busy_until_[output] = cycle_ + output_latency_ + flits - 1;
            ++branch_transmissions_;
            if (output != Local) { ++link_traversals_; flit_traversals_ += flits; }
        }
        if (outputs.size() > 1) clones_ += outputs.size() - 1;
        delete packet;
        inputs_[input].pop_front();
        auto* credit = new NocCreditV5Event();
        links_[input]->send(credit);
        round_robin_ = (input + 1) % PortCount;
    }
    return false;
}

void MulticastRouterV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::uint64_t remaining=0;
    for (const auto& queue : inputs_) remaining += queue.size();
    std::ofstream out(output_json_);
    out << "{\n  \"pe_id\": " << pe_id_
        << ",\n  \"source_packets\": " << source_packets_
        << ",\n  \"router_clones\": " << clones_
        << ",\n  \"branch_transmissions\": " << branch_transmissions_
        << ",\n  \"link_traversals\": " << link_traversals_
        << ",\n  \"flit_traversals\": " << flit_traversals_
        << ",\n  \"route_lookups\": " << route_lookups_
        << ",\n  \"credit_stall_cycles\": " << credit_stalls_
        << ",\n  \"output_stall_cycles\": " << output_stalls_
        << ",\n  \"input_queue_peak\": " << input_peak_
        << ",\n  \"remaining_packets\": " << remaining
        << ",\n  \"drops\": " << drops_ << "\n}\n";
}

void MulticastRouterV5::finish() {
    writeEvidence_();
    const auto add=[this](const char* name, std::uint64_t value) { registerStatistic<std::uint64_t>(name)->addData(value); };
    add("mcast.source_packets", source_packets_); add("mcast.router_clones", clones_);
    add("mcast.branch_transmissions", branch_transmissions_); add("mcast.link_traversals", link_traversals_);
    add("mcast.flit_traversals", flit_traversals_); add("mcast.route_lookups", route_lookups_);
    add("mcast.credit_stall_cycles", credit_stalls_); add("mcast.output_stall_cycles", output_stalls_);
}

}}}

#include <sst/core/sst_config.h>

#include "BankedSramV5Component.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
std::uint64_t positive(const SST::Params& params, const char* name, std::uint64_t fallback) {
    return std::max<std::uint64_t>(1, params.find<std::uint64_t>(name, fallback));
}
}

BankedSramV5Component::BankedSramV5Component(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.BankedSramV5", 0, 0, SST::Output::STDOUT),
      model_([&params]() {
          BankedSramV5Config config;
          config.capacity_bytes = positive(params, "capacity_bytes", 4096);
          config.banks = static_cast<std::uint32_t>(positive(params, "banks", 1));
          config.ports_per_bank = static_cast<std::uint32_t>(positive(params, "ports_per_bank", 1));
          config.interleave_bytes = positive(params, "interleave_bytes", 4);
          config.read_latency_cycles = static_cast<std::uint32_t>(positive(params, "read_latency_cycles", 1));
          config.write_latency_cycles = static_cast<std::uint32_t>(positive(params, "write_latency_cycles", 1));
          config.request_queue_entries = static_cast<std::size_t>(positive(params, "request_queue_entries", 16));
          config.response_queue_entries = static_cast<std::size_t>(positive(params, "response_queue_entries", 16));
          return config;
      }()) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    stats_json_ = params.find<std::string>("stats_json", "");
    request_link_ = configureLink("request", new SST::Event::Handler2<BankedSramV5Component, &BankedSramV5Component::handleRequest_>(this));
    response_link_ = configureLink("response");
    if (!request_link_ || !response_link_) {
        out_.fatal(CALL_INFO, -1, "BankedSramV5 requires request and response links\n");
    }
    const char* names[] = {
        "storage.sram.requests.accepted", "storage.sram.requests.retryable_rejects",
        "storage.sram.requests.capacity_rejects", "storage.sram.requests.issued",
        "storage.sram.requests.completed", "storage.sram.reads", "storage.sram.writes",
        "storage.sram.bytes_read", "storage.sram.bytes_written", "storage.sram.queue_occupancy",
        "storage.sram.response_occupancy", "storage.sram.bank_conflicts",
        "storage.sram.port_stall_cycles", "storage.sram.busy_cycles",
        "storage.sram.latency_cycles", "storage.sram.capacity_bytes", "storage.sram.resident_bytes"
    };
    for (const auto* name : names) statistics_.emplace(name, registerStatistic<std::uint64_t>(name));
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new SST::Clock::Handler2<BankedSramV5Component, &BankedSramV5Component::clockTick_>(this));
}

BankedSramV5Component::~BankedSramV5Component() = default;
void BankedSramV5Component::init(unsigned int) {}
void BankedSramV5Component::setup() {}

void BankedSramV5Component::sendResponse_(const BankedSramV5Response& response) {
    auto* event = new SramResponseEvent();
    event->request_id = response.request_id;
    event->address = response.address;
    event->service_cycle = response.service_cycle;
    event->completion_cycle = response.completion_cycle;
    event->bank = response.bank;
    event->data = response.data;
    event->accepted = response.accepted;
    event->completed = response.completed;
    event->retryable = response.retryable;
    event->reject_reason = static_cast<std::uint8_t>(response.reject);
    response_link_->send(event);
}

void BankedSramV5Component::handleRequest_(SST::Event* event) {
    auto* request = dynamic_cast<SramRequestEvent*>(event);
    if (!request) {
        delete event;
        out_.fatal(CALL_INFO, -1, "BankedSramV5 received an unexpected request event\n");
    }
    BankedSramV5Request model_request;
    model_request.request_id = request->request_id;
    model_request.address = request->address;
    model_request.data = request->data;
    model_request.write = request->write;
    BankedSramV5Response rejection;
    if (!model_.accept(model_request, current_cycle_, &rejection)) {
        sendResponse_(rejection);
    }
    delete request;
}

bool BankedSramV5Component::clockTick_(SST::Cycle_t cycle) {
    current_cycle_ = static_cast<std::uint64_t>(cycle);
    model_.tick(current_cycle_);
    for (const auto& response : model_.takeResponses()) sendResponse_(response);
    return false;
}

void BankedSramV5Component::publishStatistics_() {
    const auto& stats = model_.stats();
    const auto add = [this](const char* name, std::uint64_t value) {
        statistics_.at(name)->addData(value);
    };
    add("storage.sram.requests.accepted", stats.requests_accepted);
    add("storage.sram.requests.retryable_rejects", stats.retryable_rejects);
    add("storage.sram.requests.capacity_rejects", stats.capacity_rejects);
    add("storage.sram.requests.issued", stats.requests_issued);
    add("storage.sram.requests.completed", stats.requests_completed);
    add("storage.sram.reads", stats.reads);
    add("storage.sram.writes", stats.writes);
    add("storage.sram.bytes_read", stats.bytes_read);
    add("storage.sram.bytes_written", stats.bytes_written);
    add("storage.sram.queue_occupancy", stats.queue_occupancy_peak);
    add("storage.sram.response_occupancy", stats.response_occupancy_peak);
    add("storage.sram.bank_conflicts", stats.bank_conflicts);
    add("storage.sram.port_stall_cycles", stats.port_stall_cycles);
    add("storage.sram.busy_cycles", stats.busy_cycles);
    add("storage.sram.latency_cycles", stats.latency_cycles);
    add("storage.sram.capacity_bytes", stats.capacity_bytes);
    add("storage.sram.resident_bytes", stats.resident_bytes);
}

void BankedSramV5Component::finish() {
    publishStatistics_();
    if (model_.pending() != 0 || model_.inFlight() != 0) {
        out_.fatal(CALL_INFO, -1, "BankedSramV5 finished with outstanding requests\n");
    }
    if (!stats_json_.empty()) {
        const auto& stats = model_.stats();
        std::ofstream out(stats_json_, std::ios::out | std::ios::trunc);
        if (out.good()) {
            out << "{\n"
                << "  \"requests_accepted\": " << stats.requests_accepted << ",\n"
                << "  \"retryable_rejects\": " << stats.retryable_rejects << ",\n"
                << "  \"capacity_rejects\": " << stats.capacity_rejects << ",\n"
                << "  \"requests_issued\": " << stats.requests_issued << ",\n"
                << "  \"requests_completed\": " << stats.requests_completed << ",\n"
                << "  \"reads\": " << stats.reads << ",\n"
                << "  \"writes\": " << stats.writes << ",\n"
                << "  \"bytes_read\": " << stats.bytes_read << ",\n"
                << "  \"bytes_written\": " << stats.bytes_written << ",\n"
                << "  \"queue_occupancy_peak\": " << stats.queue_occupancy_peak << ",\n"
                << "  \"response_occupancy_peak\": " << stats.response_occupancy_peak << ",\n"
                << "  \"bank_conflicts\": " << stats.bank_conflicts << ",\n"
                << "  \"port_stall_cycles\": " << stats.port_stall_cycles << ",\n"
                << "  \"busy_cycles\": " << stats.busy_cycles << ",\n"
                << "  \"latency_cycles\": " << stats.latency_cycles << ",\n"
                << "  \"capacity_bytes\": " << stats.capacity_bytes << ",\n"
                << "  \"resident_bytes\": " << stats.resident_bytes << "\n"
                << "}\n";
        }
    }
}

} // namespace v5
} // namespace SnnDL
} // namespace SST

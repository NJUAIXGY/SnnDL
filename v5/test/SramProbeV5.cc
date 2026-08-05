#include <sst/core/sst_config.h>

#include "SramProbeV5.h"

#include <algorithm>
#include <cinttypes>
#include <fstream>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
std::uint64_t hashMix(std::uint64_t hash, std::uint64_t value) {
    return hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));
}
}

SramProbeV5::SramProbeV5(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.SramProbeV5", 0, 0, SST::Output::STDOUT),
      output_json_(params.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    request_link_ = configureLink("request");
    response_link_ = configureLink("response", new SST::Event::Handler2<SramProbeV5, &SramProbeV5::handleResponse_>(this));
    if (!request_link_ || !response_link_) {
        out_.fatal(CALL_INFO, -1, "SramProbeV5 requires request and response links\n");
    }

    // Two same-bank writes, two same-bank reads, and one request in the
    // other bank make queueing and backing correctness observable in one run.
    pending_.push_back(Request{1, 0, {0x11, 0x22, 0x33, 0x44}, true});
    pending_.push_back(Request{2, 8, {0x55, 0x66, 0x77, 0x88}, true});
    pending_.push_back(Request{3, 0, std::vector<std::uint8_t>(4, 0), false});
    pending_.push_back(Request{4, 8, std::vector<std::uint8_t>(4, 0), false});
    pending_.push_back(Request{5, 4, {0xaa, 0xbb, 0xcc, 0xdd}, true});
    pending_.push_back(Request{6, 4, std::vector<std::uint8_t>(4, 0), false});

    registerClock(params.find<std::string>("clock", "1GHz"),
                  new SST::Clock::Handler2<SramProbeV5, &SramProbeV5::clockTick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

SramProbeV5::~SramProbeV5() = default;
void SramProbeV5::init(unsigned int) {}
void SramProbeV5::setup() {}

void SramProbeV5::sendPending_() {
    // The probe intentionally offers its complete finite workload.  A retry
    // response returns the original request to pending_ without dropping it.
    while (!pending_.empty()) {
        const auto found = std::find_if(pending_.begin(), pending_.end(), [this](const Request& request) {
            // Keep the dependent read behind its write while still offering
            // all independent requests concurrently to exercise queue retry.
            return request.id != 6 || completed_ids_.count(5) != 0;
        });
        if (found == pending_.end()) break;
        const auto request = *found;
        pending_.erase(found);
        in_flight_.emplace(request.id, request);
        auto* event = new SramRequestEvent();
        event->request_id = request.id;
        event->address = request.address;
        event->data = request.data;
        event->write = request.write;
        request_link_->send(event);
    }
}

void SramProbeV5::handleResponse_(SST::Event* event) {
    auto* response = dynamic_cast<SramResponseEvent*>(event);
    if (!response) {
        delete event;
        out_.fatal(CALL_INFO, -1, "SramProbeV5 received an unexpected response event\n");
    }
    const auto found = in_flight_.find(response->request_id);
    if (found == in_flight_.end()) {
        delete response;
        out_.fatal(CALL_INFO, -1, "SramProbeV5 received an unknown response id=%" PRIu64 "\n", response->request_id);
    }
    const auto request = found->second;
    in_flight_.erase(found);
    if (!response->accepted) {
        if (!response->retryable) {
            delete response;
            out_.fatal(CALL_INFO, -1, "SramProbeV5 request was rejected permanently\n");
        }
        ++retries_;
        pending_.push_front(request);
        delete response;
        return;
    }
    if (!response->completed) {
        delete response;
        out_.fatal(CALL_INFO, -1, "SramProbeV5 received an accepted non-completion response\n");
    }
    if (!request.write) {
        if (request.id == 3 && response->data != std::vector<std::uint8_t>({0x11, 0x22, 0x33, 0x44})) {
            delete response;
            out_.fatal(CALL_INFO, -1, "SramProbeV5 readback mismatch for request 3\n");
        }
        if (request.id == 4 && response->data != std::vector<std::uint8_t>({0x55, 0x66, 0x77, 0x88})) {
            delete response;
            out_.fatal(CALL_INFO, -1, "SramProbeV5 readback mismatch for request 4\n");
        }
        if (request.id == 6 && response->data != std::vector<std::uint8_t>({0xaa, 0xbb, 0xcc, 0xdd})) {
            delete response;
            out_.fatal(CALL_INFO, -1, "SramProbeV5 readback mismatch for request 6\n");
        }
    }
    ++completed_;
    completed_ids_.insert(request.id);
    max_service_cycle_ = std::max(max_service_cycle_, response->service_cycle);
    max_completion_cycle_ = std::max(max_completion_cycle_, response->completion_cycle);
    for (const auto byte : response->data) digest_ = hashMix(digest_, byte);
    delete response;
    if (completed_ == 6 && pending_.empty() && in_flight_.empty() && !finished_) {
        finished_ = true;
        primaryComponentOKToEndSim();
    }
}

bool SramProbeV5::clockTick_(SST::Cycle_t) {
    if (!finished_) sendPending_();
    return false;
}

void SramProbeV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    out << "{\n"
        << "  \"run_class\": \"development\",\n"
        << "  \"timing_evidence\": true,\n"
        << "  \"storage_model\": \"request_driven_banked_sram\",\n"
        << "  \"completed\": " << completed_ << ",\n"
        << "  \"retryable_rejects\": " << retries_ << ",\n"
        << "  \"max_service_cycle\": " << max_service_cycle_ << ",\n"
        << "  \"max_completion_cycle\": " << max_completion_cycle_ << ",\n"
        << "  \"functional_digest\": " << digest_ << "\n"
        << "}\n";
}

void SramProbeV5::finish() {
    writeEvidence_();
    if (completed_ != 6 || !pending_.empty() || !in_flight_.empty()) {
        out_.fatal(CALL_INFO, -1, "SramProbeV5 finished before all requests completed\n");
    }
}

} // namespace v5
} // namespace SnnDL
} // namespace SST

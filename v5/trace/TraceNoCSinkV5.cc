#include <sst/core/sst_config.h>
#include "TraceNoCSinkV5.h"

#include <algorithm>
#include <fstream>

namespace SST { namespace SnnDL { namespace v5 {

TraceNoCSinkV5::TraceNoCSinkV5(SST::ComponentId_t id, SST::Params& p)
    : Component(id), out_("SnnDL.TraceNoCSinkV5", 0, 0, Output::STDOUT),
      output_json_(p.find<std::string>("output_json", "")),
      destination_pe_(p.find<std::uint32_t>("destination_pe", 0)),
      destination_core_(p.find<std::uint32_t>("destination_core", 0)),
      queue_entries_(std::max(1u, p.find<std::uint32_t>("queue_entries", 16))),
      service_latency_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("service_latency_cycles", 1))),
      drain_grace_cycles_(p.find<std::uint64_t>("drain_grace_cycles", 2)),
      expected_deliveries_(p.find<std::uint64_t>("expected_deliveries", 0)) {
    out_.setVerboseLevel(p.find<int>("verbose", 0));
    delivery_ = configureLink("delivery", new Event::Handler2<TraceNoCSinkV5, &TraceNoCSinkV5::handleDelivery_>(this));
    ack_ = configureLink("ack");
    if (!delivery_ || !ack_) out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: TraceNoCSinkV5 requires delivery and ack links\n");
    registerClock(p.find<std::string>("clock", "1GHz"), new Clock::Handler2<TraceNoCSinkV5, &TraceNoCSinkV5::tick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

TraceNoCSinkV5::~TraceNoCSinkV5() {
    for (auto& item : queue_) delete item.event;
}

void TraceNoCSinkV5::handleDelivery_(SST::Event* raw) {
    auto* event = dynamic_cast<TraceNoCDeliveryV5Event*>(raw);
    if (!event || event->format_version != TraceNoCInjectionV5Event::kFormatVersion ||
        event->destination_pe != destination_pe_ || event->destination_core != destination_core_) {
        delete raw; out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: TraceNoCSinkV5 received an invalid destination event\n");
    }
    if (queue_.size() >= queue_entries_) {
        auto* ack = new TraceNoCDeliveryAckV5Event();
        ack->event_id = event->event_id; ack->event_token = event->event_token;
        ack->destination_pe = destination_pe_; ack->destination_core = destination_core_;
        ack->accepted = false; ack->retryable = true;
        ack_->send(ack); delete event; ++retries_; return;
    }
    queue_.push_back(Pending{event, cycle_ + service_latency_});
    peak_queue_ = std::max<std::uint64_t>(peak_queue_, queue_.size());
}

bool TraceNoCSinkV5::tick_(SST::Cycle_t) {
    ++cycle_;
    if (!queue_.empty() && queue_.front().ready_cycle <= cycle_) {
        auto* event = queue_.front().event;
        queue_.pop_front();
        auto* ack = new TraceNoCDeliveryAckV5Event();
        ack->event_id = event->event_id; ack->event_token = event->event_token;
        ack->destination_pe = destination_pe_; ack->destination_core = destination_core_;
        ack->accepted = true; ack->retryable = false; ack_->send(ack);
        completed_event_ids_.push_back(event->event_id);
        delete event; ++completed_;
    }
    if ((expected_deliveries_ == 0 || completed_ >= expected_deliveries_) && queue_.empty() && !finished_) {
        if (drain_ready_cycle_ == 0) drain_ready_cycle_ = cycle_ + drain_grace_cycles_;
        if (cycle_ >= drain_ready_cycle_) {
            finished_ = true; primaryComponentOKToEndSim();
        }
    }
    return false;
}

void TraceNoCSinkV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_);
    out << "{\n  \"schema_version\": \"snndl-noc-sink-evidence/v1\",\n"
        << "  \"destination_pe\": " << destination_pe_ << ",\n"
        << "  \"destination_core\": " << destination_core_ << ",\n"
        << "  \"completed_deliveries\": " << completed_ << ",\n"
        << "  \"completed_event_ids\": [";
    for (std::size_t index = 0; index < completed_event_ids_.size(); ++index) {
        if (index != 0) out << ", ";
        out << "\"" << completed_event_ids_[index] << "\"";
    }
    out << "],\n"
        << "  \"retry_events\": " << retries_ << ",\n"
        << "  \"queue_peak\": " << peak_queue_ << ",\n"
        << "  \"queue_remaining\": " << queue_.size() << ",\n"
        << "  \"status\": \"" << ((finished_ || expected_deliveries_ == 0) ? "PASS" : "INCOMPLETE") << "\"\n}\n";
}

void TraceNoCSinkV5::finish() {
    writeEvidence_();
    if (!queue_.empty() || (expected_deliveries_ != 0 && completed_ != expected_deliveries_))
        out_.fatal(CALL_INFO, -1, "TRACE-DRAIN: TraceNoCSinkV5 finished before expected deliveries drained\n");
}

}}}

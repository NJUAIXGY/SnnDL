#include <sst/core/sst_config.h>

#include "DmaSourceV5.h"

#include <algorithm>
#include <fstream>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
template <typename T>
T positive(const SST::Params& params, const char* name, T fallback) {
    return std::max<T>(1, params.find<T>(name, fallback));
}
}

DmaSourceV5::DmaSourceV5(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.DmaSourceV5", 0, 0, SST::Output::STDOUT),
      descriptor_count_(positive<std::uint64_t>(params, "descriptors", 1)),
      source_base_(params.find<std::uint64_t>("source_base", 0)),
      destination_base_(params.find<std::uint64_t>("destination_base", 4096)),
      descriptor_bytes_(positive<std::uint64_t>(params, "descriptor_bytes", 256)),
      burst_bytes_(positive<std::uint64_t>(params, "burst_bytes", 64)),
      stride_bytes_(positive<std::uint64_t>(params, "stride_bytes", 256)),
      source_space_(static_cast<std::uint8_t>(params.find<std::uint32_t>("source_space", 0))),
      destination_space_(static_cast<std::uint8_t>(params.find<std::uint32_t>("destination_space", 4))),
      source_owner_(params.find<std::uint32_t>("source_owner", 0)),
      destination_owner_(params.find<std::uint32_t>("destination_owner", 0)),
      output_json_(params.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    descriptor_link_ = configureLink("descriptor");
    completion_link_ = configureLink(
        "completion", new SST::Event::Handler2<DmaSourceV5, &DmaSourceV5::handleCompletion_>(this));
    ready_link_ = configureLink("ready");
    if (!descriptor_link_ || !completion_link_) {
        out_.fatal(CALL_INFO, -1, "DmaSourceV5 requires descriptor and completion links\n");
    }
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new SST::Clock::Handler2<DmaSourceV5, &DmaSourceV5::clockTick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

DmaSourceV5::~DmaSourceV5() = default;

void DmaSourceV5::handleCompletion_(SST::Event* event) {
    auto* completion = dynamic_cast<DmaCompletionEvent*>(event);
    if (!completion) {
        delete event;
        out_.fatal(CALL_INFO, -1, "DmaSourceV5 received an unexpected completion event\n");
    }
    if (!completion->accepted) {
        if (!completion->retryable) {
            delete completion;
            out_.fatal(CALL_INFO, -1, "DmaSourceV5 descriptor was rejected permanently\n");
        }
        if (outstanding_.find(completion->descriptor_id) == outstanding_.end()) {
            delete completion;
            out_.fatal(CALL_INFO, -1, "DmaSourceV5 retry references an unknown descriptor\n");
        }
        retry_queue_.push_back(completion->descriptor_id);
        ++retryable_rejects_;
    } else if (completion->completed) {
        ++completed_;
        completed_bytes_ += completion->bytes;
        completed_bursts_ += completion->bursts;
        outstanding_.erase(completion->descriptor_id);
        if (completed_ >= descriptor_count_ && ready_link_ && !ready_sent_) {
            ready_link_->send(completion->clone());
            ready_sent_ = true;
        }
    }
    delete completion;
    if (completed_ >= descriptor_count_) primaryComponentOKToEndSim();
}

bool DmaSourceV5::clockTick_(SST::Cycle_t cycle) {
    current_cycle_ = static_cast<std::uint64_t>(cycle);
    if (!retry_queue_.empty()) {
        const auto descriptor_id = retry_queue_.front();
        retry_queue_.pop_front();
        const auto found = outstanding_.find(descriptor_id);
        if (found == outstanding_.end()) {
            out_.fatal(CALL_INFO, -1, "DmaSourceV5 retry queue lost descriptor\n");
        }
        sendDescriptor_(found->second);
    } else if (generated_ < descriptor_count_) {
        DmaDescriptorEvent descriptor;
        descriptor.descriptor_id = generated_ + 1;
        descriptor.timestep_id = 0;
        descriptor.source_space = source_space_;
        descriptor.destination_space = destination_space_;
        descriptor.source_owner = source_owner_;
        descriptor.destination_owner = destination_owner_;
        descriptor.source_address = source_base_ + generated_ * stride_bytes_;
        descriptor.destination_address = destination_base_ + generated_ * stride_bytes_;
        descriptor.bytes = descriptor_bytes_;
        descriptor.burst_bytes = burst_bytes_;
        descriptor.completion_token = descriptor.descriptor_id;
        outstanding_.emplace(descriptor.descriptor_id, descriptor);
        ++generated_;
        sendDescriptor_(descriptor);
    }
    return false;
}

void DmaSourceV5::sendDescriptor_(const DmaDescriptorEvent& descriptor) {
    auto* event = new DmaDescriptorEvent(descriptor);
    descriptor_link_->send(event);
    ++attempts_;
}

void DmaSourceV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    out << "{\n"
        << "  \"run_class\": \"development\",\n"
        << "  \"descriptors_generated\": " << generated_ << ",\n"
        << "  \"descriptor_attempts\": " << attempts_ << ",\n"
        << "  \"descriptors_issued\": " << attempts_ << ",\n"
        << "  \"descriptors_completed\": " << completed_ << ",\n"
        << "  \"completed_bytes\": " << completed_bytes_ << ",\n"
        << "  \"completed_bursts\": " << completed_bursts_ << ",\n"
        << "  \"retryable_rejects\": " << retryable_rejects_ << ",\n"
        << "  \"ready_sent\": " << (ready_sent_ ? "true" : "false") << ",\n"
        << "  \"last_cycle\": " << current_cycle_ << "\n"
        << "}\n";
}

void DmaSourceV5::finish() {
    if (!outstanding_.empty() || !retry_queue_.empty()) {
        out_.fatal(CALL_INFO, -1, "DmaSourceV5 finished with outstanding descriptors\n");
    }
    writeEvidence_();
}

} // namespace v5
} // namespace SnnDL
} // namespace SST

#include <sst/core/sst_config.h>

#include "SnnDmaEngineV5.h"
#include "v5/api/V5Types.h"

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
template <typename T>
T positive(const SST::Params& params, const char* name, T fallback) {
    return std::max<T>(1, params.find<T>(name, fallback));
}

std::uint64_t addChecked(std::uint64_t base, std::uint64_t offset, const char* name) {
    if (offset > std::numeric_limits<std::uint64_t>::max() - base) {
        throw std::invalid_argument(std::string("P3 DMA address overflows ") + name);
    }
    return base + offset;
}
}

SnnDmaEngineV5::SnnDmaEngineV5(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.SnnDmaEngineV5", 0, 0, SST::Output::STDOUT),
      channels_(static_cast<std::uint32_t>(positive<std::uint32_t>(params, "channels", 1))),
      queue_entries_(static_cast<std::size_t>(positive<std::uint64_t>(params, "queue_entries", 16))),
      max_outstanding_(static_cast<std::size_t>(positive<std::uint64_t>(params, "max_outstanding", 8))),
      default_burst_bytes_(positive<std::uint64_t>(params, "burst_bytes", 64)),
      bytes_per_cycle_(positive<std::uint64_t>(params, "bytes_per_cycle", 16)),
      setup_cycles_(params.find<std::uint64_t>("setup_cycles", 1)),
      expected_descriptors_(params.find<std::uint64_t>("expected_descriptors", 0)),
      stats_json_(params.find<std::string>("stats_json", "")) {
    if (max_outstanding_ > queue_entries_) {
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 max_outstanding must be <= queue_entries\n");
    }
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    descriptor_link_ = configureLink(
        "descriptor", new SST::Event::Handler2<SnnDmaEngineV5, &SnnDmaEngineV5::handleDescriptor_>(this));
    completion_link_ = configureLink("completion");
    if (!descriptor_link_ || !completion_link_) {
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 requires descriptor and completion links\n");
    }
    memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memory", ComponentInfo::SHARE_NONE, registerTimeBase("1ns"),
        new SST::Interfaces::StandardMem::Handler2<
            SnnDmaEngineV5, &SnnDmaEngineV5::handleMemory_>(this));
    if (!memory_) {
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 requires StandardMem subcomponent 'memory'\n");
    }
    const char* names[] = {
        "dma.descriptors.accepted", "dma.descriptors.completed", "dma.descriptors.retryable_rejects",
        "dma.bursts.issued", "dma.bursts.completed", "dma.bytes.issued", "dma.bytes.completed",
        "dma.queue.occupancy", "dma.outstanding.occupancy", "dma.setup_cycles", "dma.elapsed_cycles"
    };
    for (const auto* name : names) statistics_.emplace(name, registerStatistic<std::uint64_t>(name));
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new SST::Clock::Handler2<SnnDmaEngineV5, &SnnDmaEngineV5::clockTick_>(this));
    if (expected_descriptors_ != 0) {
        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();
    }
}

SnnDmaEngineV5::~SnnDmaEngineV5() = default;

void SnnDmaEngineV5::init(unsigned int phase) { memory_->init(phase); }
void SnnDmaEngineV5::setup() { memory_->setup(); }

void SnnDmaEngineV5::sendCompletion_(const DescriptorState& state, bool accepted, bool completed,
                                    bool retryable, std::uint8_t error) {
    auto* event = new DmaCompletionEvent();
    event->descriptor_id = state.descriptor.descriptor_id;
    event->timestep_id = state.descriptor.timestep_id;
    event->completion_token = state.descriptor.completion_token;
    event->bytes = completed ? state.descriptor.bytes : 0;
    event->bursts = completed ? state.burst_count : 0;
    event->elapsed_cycles = completed ? current_cycle_ - state.start_cycle : 0;
    event->accepted = accepted;
    event->completed = completed;
    event->retryable = retryable;
    event->error = error;
    completion_link_->send(event);
}

void SnnDmaEngineV5::handleDescriptor_(SST::Event* event) {
    auto* descriptor = dynamic_cast<DmaDescriptorEvent*>(event);
    if (!descriptor) {
        delete event;
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 received an unexpected descriptor event\n");
    }
    const auto id = descriptor->descriptor_id;
    const auto chip_dram = static_cast<std::uint8_t>(::SnnDL::v5::AddressSpaceId::ChipDram);
    const auto weight_spm = static_cast<std::uint8_t>(::SnnDL::v5::AddressSpaceId::PeWeightSpm);
    const bool get = descriptor->source_space == chip_dram &&
                     descriptor->destination_space == weight_spm &&
                     descriptor->source_owner == 0;
    const bool put = descriptor->source_space == weight_spm &&
                     descriptor->destination_space == chip_dram &&
                     descriptor->destination_owner == 0;
    if (!get && !put) {
        DescriptorState rejected;
        rejected.descriptor = *descriptor;
        sendCompletion_(rejected, false, false, false, 3);
        delete descriptor;
        return;
    }
    if (id == 0 || descriptor->bytes == 0 || descriptor->burst_bytes == 0 ||
        descriptors_.count(id) != 0) {
        DescriptorState rejected;
        rejected.descriptor = *descriptor;
        sendCompletion_(rejected, false, false, false, 1);
        delete descriptor;
        return;
    }
    if (order_.size() >= queue_entries_) {
        DescriptorState rejected;
        rejected.descriptor = *descriptor;
        ++descriptor_retries_;
        sendCompletion_(rejected, false, false, true, 2);
        delete descriptor;
        return;
    }
    DescriptorState state;
    state.descriptor = *descriptor;
    // A burst cannot exceed either configured burst geometry or the per-cycle
    // byte budget.  Splitting here keeps burst_count and completion accounting
    // exact while still honoring the finite byte/cycle contract.
    state.burst_bytes = std::min<std::uint64_t>(
        std::min<std::uint64_t>(descriptor->burst_bytes, default_burst_bytes_), bytes_per_cycle_);
    if (state.burst_bytes == 0) state.burst_bytes = default_burst_bytes_;
    state.burst_count = (descriptor->bytes + state.burst_bytes - 1) / state.burst_bytes;
    state.start_cycle = current_cycle_;
    state.ready_cycle = addChecked(current_cycle_, setup_cycles_, "setup cycle");
    descriptors_.emplace(id, state);
    order_.push_back(id);
    ++descriptors_accepted_;
    setup_cycles_total_ += setup_cycles_;
    queue_peak_ = std::max<std::uint64_t>(queue_peak_, order_.size());
    delete descriptor;
}

void SnnDmaEngineV5::handleMemory_(SST::Interfaces::StandardMem::Request* request) {
    if (!request) return;
    auto* response = dynamic_cast<SST::Interfaces::StandardMem::WriteResp*>(request);
    if (!response) {
        delete request;
        out_.fatal(CALL_INFO, -1,
                   "SnnDmaEngineV5 expected StandardMem::WriteResp for MoveData\n");
    }
    if (!response->getSuccess()) {
        delete request;
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 received a failed MoveData response\n");
    }
    const auto request_id = static_cast<std::uint64_t>(request->getID());
    const auto found = requests_.find(request_id);
    if (found == requests_.end()) {
        delete request;
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 received an unknown StandardMem response id=%" PRIu64 "\n", request_id);
    }
    auto descriptor = descriptors_.find(found->second.descriptor_id);
    if (descriptor == descriptors_.end()) {
        delete request;
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 response references retired descriptor\n");
    }
    ++descriptor->second.completed;
    ++bursts_completed_;
    bytes_completed_ += found->second.bytes;
    requests_.erase(found);
    delete request;
    if (descriptor->second.completed == descriptor->second.burst_count &&
        descriptor->second.issued == descriptor->second.burst_count) {
        completeDescriptor_(descriptor->first);
    }
}

void SnnDmaEngineV5::completeDescriptor_(std::uint64_t descriptor_id) {
    auto found = descriptors_.find(descriptor_id);
    if (found == descriptors_.end()) return;
    const auto state = found->second;
    sendCompletion_(state, true, true, false, 0);
    elapsed_cycles_total_ += current_cycle_ - state.start_cycle;
    ++descriptors_completed_;
    ++completed_descriptors_;
    if (!order_.empty() && order_.front() == descriptor_id) order_.pop_front();
    else {
        order_.erase(std::remove(order_.begin(), order_.end(), descriptor_id), order_.end());
    }
    descriptors_.erase(found);
    if (expected_descriptors_ != 0 && completed_descriptors_ >= expected_descriptors_ &&
        descriptors_.empty() && requests_.empty()) {
        primaryComponentOKToEndSim();
    }
}

void SnnDmaEngineV5::issueBursts_() {
    std::uint32_t issued_this_cycle = 0;
    std::uint64_t byte_budget = bytes_per_cycle_;
    while (issued_this_cycle < channels_ && requests_.size() < max_outstanding_ && !order_.empty()) {
        auto found = descriptors_.find(order_.front());
        if (found == descriptors_.end()) { order_.pop_front(); continue; }
        auto& state = found->second;
        if (state.ready_cycle > current_cycle_) break;
        if (state.next_offset >= state.descriptor.bytes) {
            if (state.completed == state.burst_count) completeDescriptor_(found->first);
            break;
        }
        const auto bytes = std::min<std::uint64_t>(
            std::min<std::uint64_t>(state.burst_bytes, state.descriptor.bytes - state.next_offset),
            byte_budget);
        if (bytes == 0) break;
        const auto source = addChecked(state.descriptor.source_address, state.next_offset, "source");
        const auto destination = addChecked(state.descriptor.destination_address, state.next_offset, "destination");
        auto* request = new SST::Interfaces::StandardMem::MoveData(source, destination, bytes);
        const auto request_id = static_cast<std::uint64_t>(request->getID());
        requests_.emplace(request_id, RequestState{found->first, bytes});
        memory_->send(request);
        state.next_offset += bytes;
        ++state.issued;
        ++bursts_issued_;
        bytes_issued_ += bytes;
        byte_budget -= bytes;
        ++issued_this_cycle;
        outstanding_peak_ = std::max<std::uint64_t>(outstanding_peak_, requests_.size());
    }
}

bool SnnDmaEngineV5::clockTick_(SST::Cycle_t cycle) {
    current_cycle_ = static_cast<std::uint64_t>(cycle);
    issueBursts_();
    return false;
}

void SnnDmaEngineV5::publishStatistics_() {
    const auto add = [this](const char* name, std::uint64_t value) {
        statistics_.at(name)->addData(value);
    };
    add("dma.descriptors.accepted", descriptors_accepted_);
    add("dma.descriptors.completed", descriptors_completed_);
    add("dma.descriptors.retryable_rejects", descriptor_retries_);
    add("dma.bursts.issued", bursts_issued_);
    add("dma.bursts.completed", bursts_completed_);
    add("dma.bytes.issued", bytes_issued_);
    add("dma.bytes.completed", bytes_completed_);
    add("dma.queue.occupancy", queue_peak_);
    add("dma.outstanding.occupancy", outstanding_peak_);
    add("dma.setup_cycles", setup_cycles_total_);
    add("dma.elapsed_cycles", elapsed_cycles_total_);
}

void SnnDmaEngineV5::writeStats_() const {
    if (stats_json_.empty()) return;
    std::ofstream out(stats_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    out << "{\n"
        << "  \"descriptors_accepted\": " << descriptors_accepted_ << ",\n"
        << "  \"descriptors_completed\": " << descriptors_completed_ << ",\n"
        << "  \"retryable_rejects\": " << descriptor_retries_ << ",\n"
        << "  \"bursts_issued\": " << bursts_issued_ << ",\n"
        << "  \"bursts_completed\": " << bursts_completed_ << ",\n"
        << "  \"bytes_issued\": " << bytes_issued_ << ",\n"
        << "  \"bytes_completed\": " << bytes_completed_ << ",\n"
        << "  \"queue_peak\": " << queue_peak_ << ",\n"
        << "  \"outstanding_peak\": " << outstanding_peak_ << ",\n"
        << "  \"setup_cycles\": " << setup_cycles_total_ << ",\n"
        << "  \"elapsed_cycles\": " << elapsed_cycles_total_ << "\n"
        << "}\n";
}

void SnnDmaEngineV5::finish() {
    memory_->finish();
    publishStatistics_();
    writeStats_();
    if (!descriptors_.empty() || !requests_.empty()) {
        out_.fatal(CALL_INFO, -1, "SnnDmaEngineV5 finished with outstanding DMA state\n");
    }
}

} // namespace v5
} // namespace SnnDL
} // namespace SST

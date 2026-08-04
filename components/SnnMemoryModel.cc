#include <sst/core/sst_config.h>

#include "SnnMemoryModel.h"

#include <algorithm>
#include <cinttypes>

namespace SST { namespace SnnDL {

extern "C" void snndl_platform2d_anchor() {}

SnnMemoryModel::SnnMemoryModel(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.SnnMemoryModel", 0, 0, SST::Output::STDOUT) {
    const auto total_pes = params.find<std::uint32_t>("total_pes", 1);
    latency_cycles_ = std::max<std::uint64_t>(1, params.find<std::uint64_t>("latency_cycles", 10));
    queue_capacity_ = std::max<std::size_t>(1, params.find<std::size_t>("queue_capacity", 4096));
    out_.setVerboseLevel(params.find<int>("verbose", 0));

    pe_links_.reserve(total_pes);
    for (std::uint32_t pe = 0; pe < total_pes; ++pe) {
        auto* link = configureLink(
            "pe" + std::to_string(pe),
            new SST::Event::Handler2<SnnMemoryModel, &SnnMemoryModel::handleRequest_>(this));
        if (!link) {
            out_.fatal(CALL_INFO, -1, "SnnMemoryModel requires port pe%u\n", pe);
        }
        pe_links_.push_back(link);
    }

    const auto clock = params.find<std::string>("clock", "1GHz");
    registerClock(clock, new SST::Clock::Handler2<SnnMemoryModel, &SnnMemoryModel::clockTick_>(this));
}

SnnMemoryModel::~SnnMemoryModel() = default;

void SnnMemoryModel::init(unsigned int) {}
void SnnMemoryModel::setup() {}

void SnnMemoryModel::finish() {
    out_.verbose(CALL_INFO, 1, 0,
                 "[snndl-memory] requests=%" PRIu64 " responses=%" PRIu64 " outstanding=%zu\n",
                 requests_, responses_, pending_.size());
}

void SnnMemoryModel::handleRequest_(SST::Event* event) {
    auto* request = dynamic_cast<MeshMemoryRequestEvent*>(event);
    if (!request) {
        delete event;
        out_.fatal(CALL_INFO, -1, "SnnMemoryModel received an unexpected event\n");
    }
    if (pending_.size() >= queue_capacity_) {
        delete request;
        out_.fatal(CALL_INFO, -1, "SnnMemoryModel request queue overflow (capacity=%zu)\n",
                   queue_capacity_);
    }

    pending_.push_back(Pending{
        static_cast<SST::Cycle_t>(getCurrentSimTimeNano() + latency_cycles_),
        request->source_pe,
        request->request_id,
        request->timestep,
        request->address,
        request->value});
    ++requests_;
    delete request;
}

bool SnnMemoryModel::clockTick_(SST::Cycle_t cycle) {
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->ready_cycle > cycle) {
            ++it;
            continue;
        }
        auto* response = new MeshMemoryResponseEvent();
        response->request_id = it->request_id;
        response->timestep = it->timestep;
        response->source_pe = it->source_pe;
        response->address = it->address;
        response->value = it->value;
        pe_links_.at(it->source_pe)->send(response);
        ++responses_;
        it = pending_.erase(it);
    }
    return false;
}

}} // namespace SST::SnnDL

#include "BankedSramV5.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
std::uint64_t endAddress(std::uint64_t address, std::size_t bytes) {
    const auto length = static_cast<std::uint64_t>(bytes);
    if (address > std::numeric_limits<std::uint64_t>::max() - length) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return address + length;
}
}

std::uint32_t BankedSramV5::positive_(std::uint32_t value) { return std::max<std::uint32_t>(1, value); }
std::uint64_t BankedSramV5::positive_(std::uint64_t value) { return std::max<std::uint64_t>(1, value); }

std::uint64_t BankedSramV5::satAdd_(std::uint64_t lhs, std::uint64_t rhs) {
    if (std::numeric_limits<std::uint64_t>::max() - lhs < rhs) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs + rhs;
}

BankedSramV5::BankedSramV5(const BankedSramV5Config& config) : config_(config) {
    config_.banks = positive_(config_.banks);
    config_.ports_per_bank = positive_(config_.ports_per_bank);
    config_.interleave_bytes = positive_(config_.interleave_bytes);
    config_.read_latency_cycles = positive_(config_.read_latency_cycles);
    config_.write_latency_cycles = positive_(config_.write_latency_cycles);
    config_.request_queue_entries = positive_(config_.request_queue_entries);
    config_.response_queue_entries = positive_(config_.response_queue_entries);
    if (config_.capacity_bytes == 0) {
        throw std::invalid_argument("P2 SRAM capacity_bytes must be positive");
    }
    if (config_.capacity_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("P2 SRAM capacity_bytes exceeds host addressable backing");
    }
    backing_.assign(static_cast<std::size_t>(config_.capacity_bytes), 0);
    port_busy_until_.assign(config_.banks, std::vector<std::uint64_t>(config_.ports_per_bank, 0));
    stats_.capacity_bytes = config_.capacity_bytes;
}

bool BankedSramV5::fits_(const BankedSramV5Request& request) const {
    if (request.data.empty()) return false;
    if (endAddress(request.address, request.data.size()) > config_.capacity_bytes) return false;
    return request.write || request.data.size() <= config_.capacity_bytes;
}

std::uint32_t BankedSramV5::bankForAddress(std::uint64_t address) const {
    return static_cast<std::uint32_t>((address / config_.interleave_bytes) % config_.banks);
}

bool BankedSramV5::accept(const BankedSramV5Request& request, std::uint64_t cycle,
                          BankedSramV5Response* rejection) {
    auto reject = [&](BankedSramV5Reject reason, bool retryable) {
        if (rejection) {
            rejection->request_id = request.request_id;
            rejection->address = request.address;
            rejection->accepted = false;
            rejection->completed = false;
            rejection->retryable = retryable;
            rejection->reject = reason;
            rejection->bank = bankForAddress(request.address);
        }
        if (reason == BankedSramV5Reject::QueueFull) ++stats_.retryable_rejects;
        if (reason == BankedSramV5Reject::Capacity) ++stats_.capacity_rejects;
        if (reason == BankedSramV5Reject::Invalid) ++stats_.invalid_rejects;
        return false;
    };

    if (request.request_id == 0 || request.data.empty() || seen_request_ids_.count(request.request_id) != 0) {
        return reject(BankedSramV5Reject::Invalid, false);
    }
    if (!fits_(request)) return reject(BankedSramV5Reject::Capacity, false);
    if (pending_.size() + in_flight_.size() >= config_.request_queue_entries ||
        pending_.size() + in_flight_.size() + responses_.size() >= config_.response_queue_entries) {
        return reject(BankedSramV5Reject::QueueFull, true);
    }

    pending_.push_back(Pending{request, bankForAddress(request.address), cycle, next_sequence_++});
    seen_request_ids_.insert(request.request_id);
    ++stats_.requests_accepted;
    stats_.queue_occupancy_peak = std::max<std::uint64_t>(
        stats_.queue_occupancy_peak, static_cast<std::uint64_t>(pending_.size() + in_flight_.size()));
    stats_.resident_bytes = std::max(stats_.resident_bytes, config_.capacity_bytes);
    return true;
}

void BankedSramV5::complete_(std::uint64_t cycle) {
    std::vector<InFlight> remaining;
    remaining.reserve(in_flight_.size());
    for (auto& item : in_flight_) {
        if (item.completion_cycle > cycle) {
            remaining.push_back(std::move(item));
            continue;
        }
        BankedSramV5Response response;
        response.request_id = item.pending.request.request_id;
        response.address = item.pending.request.address;
        response.service_cycle = item.service_cycle;
        response.completion_cycle = item.completion_cycle;
        response.bank = item.pending.bank;
        response.accepted = true;
        response.completed = true;
        response.data.resize(item.pending.request.data.size(), 0);
        const auto offset = static_cast<std::size_t>(item.pending.request.address);
        if (item.pending.request.write) {
            std::copy(item.pending.request.data.begin(), item.pending.request.data.end(), backing_.begin() + offset);
            ++stats_.writes;
            stats_.bytes_written = satAdd_(stats_.bytes_written, item.pending.request.data.size());
        } else {
            std::copy(backing_.begin() + offset,
                      backing_.begin() + offset + item.pending.request.data.size(), response.data.begin());
            ++stats_.reads;
            stats_.bytes_read = satAdd_(stats_.bytes_read, item.pending.request.data.size());
        }
        ++stats_.requests_completed;
        stats_.latency_cycles = satAdd_(stats_.latency_cycles, cycle - item.pending.enqueue_cycle);
        responses_.push_back(std::move(response));
        stats_.response_occupancy_peak = std::max<std::uint64_t>(
            stats_.response_occupancy_peak, static_cast<std::uint64_t>(responses_.size()));
    }
    in_flight_.swap(remaining);
}

void BankedSramV5::issue_(std::uint64_t cycle) {
    for (std::uint32_t bank = 0; bank < config_.banks; ++bank) {
        std::vector<std::size_t> available;
        for (std::uint32_t port = 0; port < config_.ports_per_bank; ++port) {
            if (port_busy_until_[bank][port] <= cycle) available.push_back(port);
        }
        std::size_t waiting = 0;
        for (const auto& item : pending_) if (item.bank == bank) ++waiting;
        if (waiting > available.size()) {
            ++stats_.bank_conflicts;
            ++stats_.port_stall_cycles;
        }
        for (const auto port : available) {
            auto found = std::find_if(pending_.begin(), pending_.end(),
                                      [bank](const Pending& item) { return item.bank == bank; });
            if (found == pending_.end()) break;
            const auto latency = found->request.write ? config_.write_latency_cycles : config_.read_latency_cycles;
            InFlight item{*found, cycle, cycle + latency, static_cast<std::uint32_t>(port)};
            port_busy_until_[bank][port] = item.completion_cycle;
            pending_.erase(found);
            in_flight_.push_back(std::move(item));
            ++stats_.requests_issued;
        }
        bool busy = false;
        for (const auto until : port_busy_until_[bank]) busy = busy || until > cycle;
        if (busy) ++stats_.busy_cycles;
    }
}

void BankedSramV5::tick(std::uint64_t cycle) {
    if (has_cycle_ && cycle < last_cycle_) throw std::logic_error("P2 SRAM clock moved backwards");
    has_cycle_ = true;
    last_cycle_ = cycle;
    complete_(cycle);
    issue_(cycle);
}

std::vector<BankedSramV5Response> BankedSramV5::takeResponses() {
    std::vector<BankedSramV5Response> result;
    result.reserve(responses_.size());
    while (!responses_.empty()) {
        result.push_back(std::move(responses_.front()));
        responses_.pop_front();
    }
    std::sort(result.begin(), result.end(), [](const BankedSramV5Response& lhs, const BankedSramV5Response& rhs) {
        if (lhs.completion_cycle != rhs.completion_cycle) return lhs.completion_cycle < rhs.completion_cycle;
        return lhs.request_id < rhs.request_id;
    });
    return result;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST

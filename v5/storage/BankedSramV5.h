#ifndef SST_SNN_DL_V5_BANKED_SRAM_V5_H
#define SST_SNN_DL_V5_BANKED_SRAM_V5_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <set>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

// Request-driven local storage model used by the P2 timing path.  The model
// owns the backing bytes; callers cannot observe a second, timing-free copy.
struct BankedSramV5Config {
    std::uint64_t capacity_bytes = 0;
    std::uint32_t banks = 1;
    std::uint32_t ports_per_bank = 1;
    std::uint64_t interleave_bytes = 4;
    std::uint32_t read_latency_cycles = 1;
    std::uint32_t write_latency_cycles = 1;
    std::size_t request_queue_entries = 16;
    std::size_t response_queue_entries = 16;
};

struct BankedSramV5Request {
    std::uint64_t request_id = 0;
    std::uint64_t address = 0;
    std::vector<std::uint8_t> data;
    bool write = false;
};

enum class BankedSramV5Reject : std::uint8_t {
    None = 0,
    QueueFull,
    Capacity,
    Invalid,
};

struct BankedSramV5Response {
    std::uint64_t request_id = 0;
    std::uint64_t address = 0;
    std::uint64_t service_cycle = 0;
    std::uint64_t completion_cycle = 0;
    std::uint32_t bank = 0;
    std::vector<std::uint8_t> data;
    bool accepted = false;
    bool completed = false;
    bool retryable = false;
    BankedSramV5Reject reject = BankedSramV5Reject::None;
};

struct BankedSramV5Stats {
    std::uint64_t requests_accepted = 0;
    std::uint64_t retryable_rejects = 0;
    std::uint64_t capacity_rejects = 0;
    std::uint64_t invalid_rejects = 0;
    std::uint64_t requests_issued = 0;
    std::uint64_t requests_completed = 0;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    std::uint64_t queue_occupancy_peak = 0;
    std::uint64_t response_occupancy_peak = 0;
    std::uint64_t bank_conflicts = 0;
    std::uint64_t port_stall_cycles = 0;
    std::uint64_t busy_cycles = 0;
    std::uint64_t latency_cycles = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t resident_bytes = 0;
};

class BankedSramV5 {
public:
    explicit BankedSramV5(const BankedSramV5Config& config);

    const BankedSramV5Config& config() const { return config_; }
    const BankedSramV5Stats& stats() const { return stats_; }

    // A false return never drops the request.  The reject reason tells the
    // caller whether to retry or fail closed.
    bool accept(const BankedSramV5Request& request, std::uint64_t cycle,
                BankedSramV5Response* rejection = nullptr);
    void tick(std::uint64_t cycle);
    std::vector<BankedSramV5Response> takeResponses();

    std::size_t pending() const { return pending_.size(); }
    std::size_t inFlight() const { return in_flight_.size(); }
    std::size_t responses() const { return responses_.size(); }
    std::uint32_t bankForAddress(std::uint64_t address) const;

private:
    struct Pending {
        BankedSramV5Request request;
        std::uint32_t bank = 0;
        std::uint64_t enqueue_cycle = 0;
        std::uint64_t sequence = 0;
    };
    struct InFlight {
        Pending pending;
        std::uint64_t service_cycle = 0;
        std::uint64_t completion_cycle = 0;
        std::uint32_t port = 0;
    };

    static std::uint32_t positive_(std::uint32_t value);
    static std::uint64_t positive_(std::uint64_t value);
    static std::uint64_t satAdd_(std::uint64_t lhs, std::uint64_t rhs);
    bool fits_(const BankedSramV5Request& request) const;
    void complete_(std::uint64_t cycle);
    void issue_(std::uint64_t cycle);

    BankedSramV5Config config_;
    BankedSramV5Stats stats_;
    std::vector<std::uint8_t> backing_;
    std::deque<Pending> pending_;
    std::vector<InFlight> in_flight_;
    std::deque<BankedSramV5Response> responses_;
    std::set<std::uint64_t> seen_request_ids_;
    std::vector<std::vector<std::uint64_t>> port_busy_until_;
    std::uint64_t next_sequence_ = 0;
    std::uint64_t last_cycle_ = 0;
    bool has_cycle_ = false;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif

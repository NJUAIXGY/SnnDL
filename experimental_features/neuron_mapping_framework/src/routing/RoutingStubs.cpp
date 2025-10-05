/**
 * @file RoutingStubs.cpp
 * @brief 路由系统最小可编译实现（与当前头文件保持一致）
 *
 * 说明：
 * - 提供 AddressEvent / AddressEventBatch / RoutingTable / SpikeRouter / RoutingTableGenerator
 *   的基础实现以保证 src/ 编译通过；算法采用简化逻辑，便于后续逐步增强。
 */

#include "routing/AddressEvent.h"
#include "routing/RoutingTable.h"
#include "routing/SpikeRouter.h"
#include "routing/RoutingTableGenerator.h"
#include "routing/MulticastGroup.h"
#include "core/MappingSolution.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cmath>

namespace NeuronMapping {

// ========================= AddressEvent =========================

AddressEvent::AddressEvent(GlobalNeuronId global_id,
                           Timestamp ts,
                           float weight,
                           Priority priority)
    : global_id_(global_id), timestamp_(ts), weight_(weight), priority_(priority) {}

AddressEvent::AddressEvent(GlobalNeuronId global_id,
                           Timestamp ts,
                           EventType type,
                           float weight,
                           Priority priority)
    : global_id_(global_id), timestamp_(ts), weight_(weight),
      event_type_(type), priority_(priority) {}

uint32_t AddressEvent::generateRoutingMask(uint8_t mask_bits) const {
    if (mask_bits >= 32) return 0xFFFFFFFFu;
    if (mask_bits == 0) return 0u;
    // 高位匹配：例如 mask_bits=16 -> 0xFFFF0000
    return 0xFFFFFFFFu << (32 - mask_bits);
}

bool AddressEvent::matchesRoute(uint32_t key, uint32_t mask) const {
    return (getRoutingKey() & mask) == (key & mask);
}

void AddressEvent::setMulticastGroup(uint16_t group_id) {
    reserved_ = group_id; // 复用 reserved_ 存组ID
    event_type_ = EventType::MULTICAST;
}

uint16_t AddressEvent::getMulticastGroup() const { return reserved_; }

bool AddressEvent::canBatchWith(const AddressEvent& other, uint64_t time_window) const {
    return global_id_ == other.global_id_ &&
           (timestamp_ > other.timestamp_ ? (timestamp_ - other.timestamp_) : (other.timestamp_ - timestamp_)) <= time_window;
}

bool AddressEvent::mergeWith(const AddressEvent& other) {
    if (!canBatchWith(other)) return false;
    weight_ += other.weight_;
    timestamp_ = std::max(timestamp_, other.timestamp_);
    return true;
}

std::vector<uint8_t> AddressEvent::serialize() const {
    // 简单固定长度序列化：global_id(4) | timestamp(8) | weight(4) | event_type(1) | priority(1) | seq(4) | flags(2) | reserved(2)
    std::vector<uint8_t> buf;
    buf.resize(SERIALIZED_SIZE, 0);
    auto wr32 = [&](size_t off, uint32_t v){ std::memcpy(&buf[off], &v, sizeof(uint32_t)); };
    auto wr64 = [&](size_t off, uint64_t v){ std::memcpy(&buf[off], &v, sizeof(uint64_t)); };
    wr32(0, global_id_);
    wr64(4, timestamp_);
    std::memcpy(&buf[12], &weight_, sizeof(float));
    buf[16] = static_cast<uint8_t>(event_type_);
    buf[17] = static_cast<uint8_t>(priority_);
    wr32(18, sequence_number_);
    std::memcpy(&buf[22], &flags_, sizeof(uint16_t));
    std::memcpy(&buf[22 + sizeof(uint16_t)], &reserved_, sizeof(uint16_t));
    return buf;
}

bool AddressEvent::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < SERIALIZED_SIZE) return false;
    auto rd32 = [&](size_t off){ uint32_t v; std::memcpy(&v, &data[off], sizeof(uint32_t)); return v; };
    auto rd64 = [&](size_t off){ uint64_t v; std::memcpy(&v, &data[off], sizeof(uint64_t)); return v; };
    global_id_ = rd32(0);
    timestamp_ = rd64(4);
    std::memcpy(&weight_, &data[12], sizeof(float));
    event_type_ = static_cast<EventType>(data[16]);
    priority_ = static_cast<Priority>(data[17]);
    sequence_number_ = rd32(18);
    std::memcpy(&flags_, &data[22], sizeof(uint16_t));
    std::memcpy(&reserved_, &data[24], sizeof(uint16_t));
    return true;
}

bool AddressEvent::operator==(const AddressEvent& other) const {
    return global_id_ == other.global_id_ && timestamp_ == other.timestamp_ &&
           event_type_ == other.event_type_ && priority_ == other.priority_ &&
           weight_ == other.weight_;
}

bool AddressEvent::operator<(const AddressEvent& other) const {
    if (timestamp_ != other.timestamp_) return timestamp_ < other.timestamp_;
    if (priority_ != other.priority_) return priority_ > other.priority_; // 高优先级靠前
    return global_id_ < other.global_id_;
}

bool AddressEvent::isValid() const {
    return global_id_ != INVALID_GLOBAL_ID;
}

std::string AddressEvent::toString() const {
    std::ostringstream oss;
    oss << "AER{gid=0x" << std::hex << global_id_ << std::dec
        << ", ts=" << timestamp_ << ", w=" << weight_ << "}";
    return oss.str();
}

void AddressEvent::reset() {
    global_id_ = INVALID_GLOBAL_ID;
    timestamp_ = 0;
    weight_ = 1.0f;
    event_type_ = EventType::SPIKE;
    priority_ = PRIORITY_NORMAL;
    sequence_number_ = 0;
    flags_ = 0;
    reserved_ = 0;
}

AddressEvent AddressEvent::clone() const { return *this; }

AddressEvent::GlobalNeuronId AddressEvent::generateGlobalId(NeuronId local_id, PEId pe_id, uint16_t core_id) {
    // 位布局：local(16) | pe(12) | core(4)
    uint32_t lid = static_cast<uint32_t>(local_id) & LOCAL_ID_MASK;
    uint32_t pid = static_cast<uint32_t>(pe_id) & PE_ID_MASK;
    uint32_t cid = static_cast<uint32_t>(core_id) & CORE_ID_MASK;
    return (lid) | (pid << LOCAL_ID_BITS) | (cid << (LOCAL_ID_BITS + PE_ID_BITS));
}

std::tuple<NeuronId, PEId, uint16_t> AddressEvent::extractLocalInfo(GlobalNeuronId gid) {
    NeuronId lid = static_cast<NeuronId>(gid & LOCAL_ID_MASK);
    PEId pid = static_cast<PEId>((gid >> LOCAL_ID_BITS) & PE_ID_MASK);
    uint16_t cid = static_cast<uint16_t>((gid >> (LOCAL_ID_BITS + PE_ID_BITS)) & CORE_ID_MASK);
    return {lid, pid, cid};
}

AddressEvent::Timestamp AddressEvent::getCurrentTimestamp() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

AddressEvent AddressEvent::createControlEvent(uint32_t command, Timestamp ts) {
    AddressEvent evt(command, ts, EventType::CONTROL);
    return evt;
}

AddressEvent AddressEvent::createHeartbeatEvent(PEId pe_id, Timestamp ts) {
    AddressEvent evt(static_cast<GlobalNeuronId>(pe_id), ts, EventType::HEARTBEAT);
    return evt;
}

void AddressEvent::validateFields() const {
    // 简化校验：仅检查ID有效
    (void)isValid();
}

uint32_t AddressEvent::calculateChecksum() const {
    // 简单异或校验
    return static_cast<uint32_t>(global_id_ ^ static_cast<uint32_t>(timestamp_) ^ sequence_number_);
}

// ====================== AddressEventBatch =======================

AddressEventBatch::AddressEventBatch(size_t reserve_size) { events_.reserve(reserve_size); }

void AddressEventBatch::addEvent(const AddressEvent& e) { events_.push_back(e); invalidateChecksum(); }
void AddressEventBatch::addEvent(AddressEvent&& e) { events_.push_back(std::move(e)); invalidateChecksum(); }
void AddressEventBatch::addEvents(const std::vector<AddressEvent>& es) { events_.insert(events_.end(), es.begin(), es.end()); invalidateChecksum(); }

void AddressEventBatch::removeEvent(size_t idx) {
    if (idx < events_.size()) { events_.erase(events_.begin() + idx); invalidateChecksum(); }
}
void AddressEventBatch::clear() { events_.clear(); invalidateChecksum(); }

void AddressEventBatch::sortByTimestamp() { std::sort(events_.begin(), events_.end(), [](const AddressEvent& a, const AddressEvent& b){ return a.getTimestamp() < b.getTimestamp(); }); }
void AddressEventBatch::sortByPriority() { std::sort(events_.begin(), events_.end(), [](const AddressEvent& a, const AddressEvent& b){ return a.getPriority() > b.getPriority(); }); }

size_t AddressEventBatch::mergeEvents(uint64_t time_window) {
    if (events_.empty()) return 0;
    sortByTimestamp();
    size_t merged = 0;
    std::vector<AddressEvent> out;
    out.reserve(events_.size());
    AddressEvent cur = events_[0];
    for (size_t i = 1; i < events_.size(); ++i) {
        if (!cur.mergeWith(events_[i])) { out.push_back(cur); cur = events_[i]; }
        else { merged++; }
    }
    out.push_back(cur);
    events_.swap(out);
    invalidateChecksum();
    return merged;
}

std::vector<AddressEventBatch> AddressEventBatch::split(size_t max_batch_size) const {
    std::vector<AddressEventBatch> res;
    if (max_batch_size == 0) return res;
    for (size_t i = 0; i < events_.size(); i += max_batch_size) {
        AddressEventBatch b;
        size_t end = std::min(events_.size(), i + max_batch_size);
        b.events_.insert(b.events_.end(), events_.begin() + i, events_.begin() + end);
        res.push_back(std::move(b));
    }
    return res;
}

std::vector<uint8_t> AddressEventBatch::serialize() const {
    std::vector<uint8_t> buf;
    for (const auto& e : events_) {
        auto v = e.serialize();
        buf.insert(buf.end(), v.begin(), v.end());
    }
    return buf;
}

bool AddressEventBatch::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() % AddressEvent::getSerializedSize() != 0) return false;
    events_.clear();
    size_t sz = AddressEvent::getSerializedSize();
    for (size_t off = 0; off < data.size(); off += sz) {
        AddressEvent e;
        std::vector<uint8_t> slice(data.begin() + off, data.begin() + off + sz);
        if (!e.deserialize(slice)) return false;
        events_.push_back(std::move(e));
    }
    invalidateChecksum();
    return true;
}

std::string AddressEventBatch::getStatistics() const {
    std::ostringstream oss;
    oss << "batch_size=" << events_.size();
    return oss.str();
}

size_t AddressEventBatch::calculateChecksum() const {
    size_t sum = 0;
    for (const auto& e : events_) sum ^= e.getRoutingKey();
    return sum;
}

// ========================= RoutingTable =========================

static inline uint8_t maskBits(uint32_t mask) {
    // 统计高位连续1的个数（简化实现）
    if (mask == 0) return 0;
    uint8_t bits = 0;
    for (int i = 31; i >= 0; --i) {
        if ((mask >> i) & 1u) bits++; else break;
    }
    return bits;
}

uint8_t RoutingEntry::getMaskBits() const { return maskBits(mask); }

std::string RoutingEntry::toString() const {
    std::ostringstream oss;
    oss << "{key=0x" << std::hex << key << ", mask=0x" << mask << std::dec
        << ", routes=" << routes.size() << ", prio=" << priority << "}";
    return oss.str();
}

RoutingTable::RoutingTable(PEId pe_id) : pe_id_(pe_id) {}

bool RoutingTable::addEntry(const RoutingEntry& entry) {
    entries_.push_back(entry);
    invalidateIndex();
    return true;
}

bool RoutingTable::addUnicastRoute(uint32_t key, uint32_t mask, RouteDirection dir, uint16_t priority) {
    RoutingEntry e(key, mask, std::vector<RouteDirection>{dir});
    e.priority = priority;
    return addEntry(e);
}

bool RoutingTable::addMulticastRoute(uint32_t key, uint32_t mask, const std::vector<RouteDirection>& dirs, uint16_t priority) {
    RoutingEntry e(key, mask, dirs);
    e.priority = priority;
    return addEntry(e);
}

bool RoutingTable::removeEntry(uint32_t key, uint32_t mask) {
    auto sz_before = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const RoutingEntry& e){ return e.key == key && e.mask == mask; }), entries_.end());
    invalidateIndex();
    return entries_.size() != sz_before;
}

void RoutingTable::clear() { entries_.clear(); invalidateIndex(); }

std::vector<RouteDirection> RoutingTable::lookupRoute(uint32_t packet_key) const {
    if (const CacheEntry* ce = findCacheEntry(packet_key)) { updateLookupStats(true, true); return ce->routes; }
    const RoutingEntry* best = findBestMatch(packet_key);
    if (!best) { updateLookupStats(false, false); return {}; }
    updateLookupStats(true, false);
    updateCacheEntry(packet_key, best->routes);
    return best->routes;
}

const RoutingEntry* RoutingTable::findBestMatch(uint32_t packet_key) const {
    const RoutingEntry* best = nullptr;
    uint8_t best_bits = 0;
    for (const auto& e : entries_) {
        if (e.matches(packet_key)) {
            uint8_t bits = e.getMaskBits();
            if (!best || bits > best_bits || (bits == best_bits && e.priority > best->priority)) {
                best = &e; best_bits = bits;
            }
        }
    }
    return best;
}

std::vector<const RoutingEntry*> RoutingTable::findAllMatches(uint32_t packet_key) const {
    std::vector<const RoutingEntry*> res;
    for (const auto& e : entries_) if (e.matches(packet_key)) res.push_back(&e);
    return res;
}

bool RoutingTable::hasRoute(uint32_t packet_key) const { return findBestMatch(packet_key) != nullptr; }

bool RoutingTable::addMulticastGroup(const MulticastGroup& group) {
    multicast_groups_[group.group_id] = group;
    return true;
}

bool RoutingTable::removeMulticastGroup(uint16_t group_id) { return multicast_groups_.erase(group_id) > 0; }

const MulticastGroup* RoutingTable::findMulticastGroup(uint16_t group_id) const {
    auto it = multicast_groups_.find(group_id);
    return it == multicast_groups_.end() ? nullptr : &it->second;
}

std::vector<MulticastGroup> RoutingTable::getAllMulticastGroups() const {
    std::vector<MulticastGroup> v;
    v.reserve(multicast_groups_.size());
    for (const auto& kv : multicast_groups_) v.push_back(kv.second);
    return v;
}

size_t RoutingTable::compressTable() { return 0; }
void RoutingTable::optimizeEntryOrder() { std::sort(entries_.begin(), entries_.end(), EntryComparator{}); }

std::vector<std::string> RoutingTable::validateTable() const {
    std::vector<std::string> errs;
    // 检查重复 key/mask
    for (size_t i = 0; i < entries_.size(); ++i) {
        for (size_t j = i + 1; j < entries_.size(); ++j) {
            if (entries_[i].key == entries_[j].key && entries_[i].mask == entries_[j].mask) {
                errs.push_back("duplicate entry for key/mask");
            }
        }
        if (entries_[i].routes.empty()) errs.push_back("entry with empty routes");
    }
    return errs;
}

size_t RoutingTable::removeUnusedEntries(uint32_t min_packet_count) {
    auto before = entries_.size();
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const RoutingEntry& e){ return e.packet_count <= min_packet_count; }), entries_.end());
    return before - entries_.size();
}

std::unordered_map<std::string, float> RoutingTable::getUtilizationStats() const {
    return {{"entries", static_cast<float>(entries_.size())}};
}

std::string RoutingTable::getStatistics() const {
    std::ostringstream oss;
    oss << "entries=" << entries_.size();
    return oss.str();
}

void RoutingTable::resetStatistics() { total_lookups_ = successful_lookups_ = cache_hits_ = 0; }

std::string RoutingTable::exportTable(const std::string& format) const {
    std::ostringstream oss;
    if (format == "csv") {
        oss << "key,mask,routes,priority\n";
        for (const auto& e : entries_) {
            oss << e.key << "," << e.mask << "," << e.routes.size() << "," << e.priority << "\n";
        }
    } else { // json (简化)
        oss << "{";
        oss << "\"entries\":[";
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& e = entries_[i];
            oss << "{\"key\":" << e.key << ",\"mask\":" << e.mask << ",\"priority\":" << e.priority << ",\"routes\":[";
            for (size_t j = 0; j < e.routes.size(); ++j) {
                oss << static_cast<int>(e.routes[j]);
                if (j + 1 < e.routes.size()) oss << ",";
            }
            oss << "]}";
            if (i + 1 < entries_.size()) oss << ",";
        }
        oss << "]}";
    }
    return oss.str();
}

bool RoutingTable::importTable(const std::string&, const std::string&) { return false; }

std::vector<uint8_t> RoutingTable::serializeBinary() const {
    std::vector<uint8_t> out;
    for (const auto& e : entries_) {
        uint32_t key = e.key, mask = e.mask; uint16_t pr = e.priority; uint16_t n = static_cast<uint16_t>(e.routes.size());
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&key), reinterpret_cast<uint8_t*>(&key) + 4);
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&mask), reinterpret_cast<uint8_t*>(&mask) + 4);
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&pr), reinterpret_cast<uint8_t*>(&pr) + 2);
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&n), reinterpret_cast<uint8_t*>(&n) + 2);
        for (auto d : e.routes) {
            uint8_t v = static_cast<uint8_t>(d);
            out.push_back(v);
        }
    }
    return out;
}

bool RoutingTable::deserializeBinary(const std::vector<uint8_t>&) { return false; }

void RoutingTable::printTable(size_t max_entries) const {
    size_t cnt = std::min(max_entries, entries_.size());
    for (size_t i = 0; i < cnt; ++i) {
        (void)entries_[i]; // 可扩展为实际打印
    }
}

std::string RoutingTable::getDetailedInfo() const { return getStatistics(); }

std::string RoutingTable::validateRoute(uint32_t packet_key) const {
    auto routes = lookupRoute(packet_key);
    std::ostringstream oss;
    oss << (routes.empty() ? "no-route" : "ok");
    return oss.str();
}

void RoutingTable::buildIndex() const {
    key_index_.clear();
    for (size_t i = 0; i < entries_.size(); ++i) key_index_[entries_[i].key].push_back(i);
    index_valid_ = true;
}

void RoutingTable::updateLookupStats(bool success, bool cache_hit) const {
    ++total_lookups_;
    if (success) ++successful_lookups_;
    if (cache_hit) ++cache_hits_;
}

void RoutingTable::updateCacheEntry(uint32_t key, const std::vector<RouteDirection>& routes) const {
    if (lookup_cache_.size() >= CACHE_SIZE) lookup_cache_.erase(lookup_cache_.begin());
    lookup_cache_.push_back({key, routes, 0});
}

const RoutingTable::CacheEntry* RoutingTable::findCacheEntry(uint32_t key) const {
    for (const auto& c : lookup_cache_) if (c.key == key) return &c;
    return nullptr;
}

bool RoutingTable::EntryComparator::operator()(const RoutingEntry& a, const RoutingEntry& b) const {
    if (a.priority != b.priority) return a.priority > b.priority;
    if (a.getMaskBits() != b.getMaskBits()) return a.getMaskBits() > b.getMaskBits();
    return a.key < b.key;
}

bool RoutingTable::canMergeEntries(const RoutingEntry& a, const RoutingEntry& b) const {
    return a.mask == b.mask && a.routes == b.routes; // 简化：相同掩码且路由完全相同
}

RoutingEntry RoutingTable::mergeEntries(const RoutingEntry& a, const RoutingEntry& b) const {
    RoutingEntry r = a; (void)b; return r; // 简化占位
}

// ========================= SpikeRouter ==========================

SpikeRouter::SpikeRouter(const RouterConfig& config) : config_(config) {}
SpikeRouter::~SpikeRouter() = default;

bool SpikeRouter::initialize(const HardwareTopology&) { return true; }

void SpikeRouter::setRoutingTable(std::unique_ptr<RoutingTable> table) { routing_table_ = std::move(table); }
// getRoutingTable() 已在头文件内联

bool SpikeRouter::configurePort(const RouterPort& port) { ports_[port.direction] = port; return true; }
const RouterPort* SpikeRouter::getPort(RouteDirection dir) const { auto it = ports_.find(dir); return it==ports_.end()?nullptr:&it->second; }

bool SpikeRouter::routePacket(const AddressEvent& packet, RouteDirection) {
    if (!routing_table_) return false;
    auto routes = routing_table_->lookupRoute(packet.getRoutingKey());
    // 仅更新统计，实际发送留给外部系统
    (void)routes;
    return !routes.empty();
}

size_t SpikeRouter::routePacketBatch(const AddressEventBatch& batch, RouteDirection in) {
    size_t ok = 0; for (const auto& e : batch) if (routePacket(e, in)) ++ok; return ok;
}

size_t SpikeRouter::routeMulticastPacket(const AddressEvent& packet, RouteDirection in) {
    return routePacket(packet, in) ? 1 : 0;
}

std::vector<RouteDirection> SpikeRouter::camLookup(uint32_t key) const {
    if (!routing_table_) return {};
    return routing_table_->lookupRoute(key);
}

std::vector<std::vector<RouteDirection>> SpikeRouter::parallelCamLookup(const std::vector<uint32_t>& keys) const {
    std::vector<std::vector<RouteDirection>> res; res.reserve(keys.size());
    for (auto k : keys) res.push_back(camLookup(k));
    return res;
}

std::vector<RouteDirection> SpikeRouter::priorityCamLookup(uint32_t key, uint16_t) const { return camLookup(key); }

float SpikeRouter::getPortCongestion(RouteDirection) const { return 0.0f; }
void SpikeRouter::applyTrafficShaping(RouteDirection, uint32_t) {}
RouterStatistics SpikeRouter::getStatistics() const { return {}; }
void SpikeRouter::resetStatistics() {}
std::unordered_map<std::string, uint64_t> SpikeRouter::getPortStatistics(RouteDirection) const { return {}; }
void SpikeRouter::triggerPacketEvent(const AddressEvent&, RouteDirection) {}
void SpikeRouter::triggerErrorEvent(const std::string&, const AddressEvent&) {}
bool SpikeRouter::addRoutingEntry(const RoutingEntry& e) { return routing_table_ ? routing_table_->addEntry(e) : false; }
bool SpikeRouter::removeRoutingEntry(uint32_t k, uint32_t m) { return routing_table_ ? routing_table_->removeEntry(k,m) : false; }
bool SpikeRouter::updateRoutingEntry(uint32_t k, uint32_t m, const std::vector<RouteDirection>& r) { return routing_table_ ? routing_table_->addMulticastRoute(k,m,r) : false; }
size_t SpikeRouter::batchUpdateRoutingTable(const std::vector<RoutingEntry>& es) { size_t n=0; if (!routing_table_) return 0; for (const auto& e:es) if (routing_table_->addEntry(e)) ++n; return n; }
bool SpikeRouter::detectPortFailure(RouteDirection) { return false; }
void SpikeRouter::markPortFailure(RouteDirection, bool) {}
size_t SpikeRouter::rerouteFailedTraffic(RouteDirection) { return 0; }
std::string SpikeRouter::getStatus() const { return "ok"; }
std::vector<std::string> SpikeRouter::validateConfiguration() const { return {}; }
std::string SpikeRouter::generateDiagnosticReport() const { return ""; }
void SpikeRouter::enableDebugMode(bool, uint8_t) {}
void SpikeRouter::optimizeRoutingCache() {}
void SpikeRouter::preloadRoutingEntries(const std::vector<uint32_t>&) {}
void SpikeRouter::updateRouteCache(uint32_t, const std::vector<RouteDirection>&) const {}
const std::vector<RouteDirection>* SpikeRouter::findCachedRoute(uint32_t) const { return nullptr; }
void SpikeRouter::cleanupRouteCache() const {}
bool SpikeRouter::isPortHealthy(RouteDirection) const { return true; }
void SpikeRouter::performHealthCheck() {}
std::vector<RouteDirection> SpikeRouter::findAlternativeRoutes(RouteDirection, uint32_t) const { return {}; }
void SpikeRouter::initializeBuffers() {}
void SpikeRouter::initializePorts(const HardwareTopology&) {}
bool SpikeRouter::validatePacket(const AddressEvent&) const { return true; }
void SpikeRouter::updateStatistics(const AddressEvent&, bool, std::chrono::nanoseconds) {}
void SpikeRouter::updatePortStatistics(RouteDirection, const std::string&, uint64_t) {}
void SpikeRouter::applyBackpressure(RouteDirection) {}
void SpikeRouter::logDebugMessage(const std::string&) const {}

// ===================== RoutingTableGenerator ====================

RoutingTableGenerator::RoutingTableGenerator(const RoutingGenerationConfig& cfg) : config_(cfg) {}

// 生成分布式路由表（阶段一：最短路单播 + 前缀聚合压缩）
std::unordered_map<PEId, std::unique_ptr<RoutingTable>> RoutingTableGenerator::generateRoutingTables(
    const NeuralNetwork& network, const HardwareTopology& topology, const MappingSolution& mapping, const RoutingGenerationConfig& cfg) {
    config_ = cfg;

    // 统一收集待生成的表项（便于去重与前缀聚合）
    struct EntrySpec { uint32_t key; uint32_t mask; std::vector<RouteDirection> routes; uint16_t priority; };
    std::unordered_map<PEId, std::vector<EntrySpec>> pending;

    // local_idx 将在 1.2) 路由模式分组后生成
    std::unordered_map<NeuronId, uint16_t> local_idx;

    // 1) 构建源神经元的扇出表
    std::unordered_map<NeuronId, std::vector<NeuronId>> fanouts;
    for (const auto& conn : network.getAllConnections()) {
        fanouts[conn.source_id].push_back(conn.target_id);
    }

    // 1.1) 预计算各源在源PE的出向路由模式（first-hop directions 集合），用于 local_id 分组重排
    struct RoutesKey {
        std::vector<RouteDirection> dirs; // 排序去重后的方向集合
        bool operator==(const RoutesKey& o) const { return dirs == o.dirs; }
    };
    struct RKHash { size_t operator()(RoutesKey const& rk) const { size_t h=0; for(auto d:rk.dirs) h = h*131 + static_cast<size_t>(d); return h; } };

    std::unordered_map<NeuronId, RoutesKey> src_route_key;
    for (const auto& [src_neuron, targets] : fanouts) {
        PEId src_pe = mapping.getNeuronPE(src_neuron);
        if (src_pe == INVALID_PE_ID) continue;
        // 目标PE集合
        std::vector<PEId> target_pes;
        target_pes.reserve(targets.size());
        for (auto t : targets) { PEId pe = mapping.getNeuronPE(t); if (pe != INVALID_PE_ID) target_pes.push_back(pe); }
        std::sort(target_pes.begin(), target_pes.end());
        target_pes.erase(std::unique(target_pes.begin(), target_pes.end()), target_pes.end());
        bool use_multicast = config_.enable_multicast && target_pes.size() >= 3;

        std::vector<RouteDirection> first_hops;
        if (use_multicast) {
            // 汇总到每个目标的第一跳方向
            std::unordered_set<RouteDirection> set;
            for (auto dst_pe : target_pes) {
                auto path = topology.getPath(src_pe, dst_pe);
                if (path.size() >= 2) {
                    set.insert(getRouteDirection(src_pe, path[1], topology));
                } else {
                    set.insert(RouteDirection::LOCAL);
                }
            }
            first_hops.assign(set.begin(), set.end());
        } else {
            // 单播：对每个目标的第一跳方向去重
            std::unordered_set<RouteDirection> set;
            for (auto t : targets) {
                PEId dst_pe = mapping.getNeuronPE(t);
                if (dst_pe == INVALID_PE_ID) continue;
                auto path = topology.getPath(src_pe, dst_pe);
                if (path.size() >= 2) set.insert(getRouteDirection(src_pe, path[1], topology));
                else set.insert(RouteDirection::LOCAL);
            }
            first_hops.assign(set.begin(), set.end());
        }
        std::sort(first_hops.begin(), first_hops.end());
        first_hops.erase(std::unique(first_hops.begin(), first_hops.end()), first_hops.end());
        src_route_key[src_neuron] = RoutesKey{first_hops};
    }

    // 1.2) 基于(PE, core, routes_key)做local_id分组重排（每核≤128），提升连续LSB前缀聚合命中
    // 注意：local_idx在前已声明
    {
        // 聚合每核的分组
        struct GroupKey { PEId pe; uint16_t core; RoutesKey rk; };
        struct GKHash { size_t operator()(GroupKey const& g) const { RKHash rh; return (static_cast<size_t>(g.pe)*131 + g.core)*131 + rh(g.rk); } };
        struct GKEqual { bool operator()(GroupKey const& a, GroupKey const& b) const { return a.pe==b.pe && a.core==b.core && a.rk==b.rk; } };

        // 收集每(PE,core)全部神经元
        std::unordered_map<PEId, std::unordered_map<uint16_t, std::vector<NeuronId>>> per_core;
        for (const auto& asg : mapping.getAllAssignments()) per_core[asg.pe_id][static_cast<uint16_t>(asg.core_id)].push_back(asg.neuron_id);

        for (auto& pkv : per_core) {
            PEId pe = pkv.first;
            for (auto& ckv : pkv.second) {
                uint16_t core = ckv.first;
                auto& vec = ckv.second;
                // 分组：同routes_key的源挤在一起
                std::unordered_map<RoutesKey, std::vector<NeuronId>, RKHash> groups;
                for (auto nid : vec) {
                    auto it = src_route_key.find(nid);
                    RoutesKey rk;
                    if (it != src_route_key.end()) rk = it->second; // 部分无扇出时为空集合
                    groups[rk].push_back(nid);
                }
                // 大组优先，分配对齐块
                // 收集组列表
                struct Item { RoutesKey rk; std::vector<NeuronId>* list; };
                std::vector<Item> items; items.reserve(groups.size());
                for (auto& gkv : groups) items.push_back(Item{gkv.first, &gkv.second});
                std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){ return a.list->size() > b.list->size(); });

                uint32_t pos = 0;
                auto align_up = [](uint32_t x, uint32_t p){ return (p==0)?x: ((x + p - 1) / p) * p; };

                // 先分配按最大2^t块
                for (auto& it2 : items) {
                    auto& lst = *it2.list;
                    // 稳定排序避免抖动
                    std::sort(lst.begin(), lst.end());
                    size_t remaining = lst.size();
                    size_t idx = 0;
                    while (remaining > 0 && pos < 128u) {
                        // 本次块大小
                        uint32_t blk = 1u << static_cast<uint32_t>(std::floor(std::log2(std::min<size_t>(remaining, 128u - pos))));
                        // 对齐到blk边界
                        uint32_t aligned = align_up(pos, blk);
                        if (aligned + blk > 128u) { break; }
                        // 分配blk个local_id
                        for (uint32_t k = 0; k < blk; ++k) {
                            local_idx[lst[idx + k]] = static_cast<uint16_t>(aligned + k);
                        }
                        idx += blk; remaining -= blk; pos = aligned + blk;
                    }
                    // 对于未分配的（极少数），顺序塞入余下位置
                    while (remaining > 0 && pos < 128u) {
                        local_idx[lst[idx]] = static_cast<uint16_t>(pos);
                        ++idx; --remaining; ++pos;
                    }
                }
            }
        }
    }

    // 2) 针对每个源，走最短路单播或多播树
    for (const auto& [src_neuron, targets] : fanouts) {
        PEId src_pe = mapping.getNeuronPE(src_neuron);
        if (src_pe == INVALID_PE_ID) continue;
        uint16_t src_core = static_cast<uint16_t>(mapping.getNeuronCore(src_neuron));
        uint16_t lid = 0;
        auto it_l = local_idx.find(src_neuron);
        if (it_l != local_idx.end()) lid = it_l->second;
        uint32_t key = AddressEvent::generateGlobalId(lid, src_pe, src_core);

        // 目标PE集合
        std::vector<PEId> target_pes;
        target_pes.reserve(targets.size());
        for (auto t : targets) {
            PEId pe = mapping.getNeuronPE(t);
            if (pe != INVALID_PE_ID) target_pes.push_back(pe);
        }
        std::sort(target_pes.begin(), target_pes.end());
        target_pes.erase(std::unique(target_pes.begin(), target_pes.end()), target_pes.end());

        bool use_multicast = config_.enable_multicast && target_pes.size() >= 3;

        if (use_multicast) {
            // 构建“最短路径树”的并集（近似Steiner）
            std::unordered_map<PEId, std::unordered_set<PEId>> children;
            for (auto dst_pe : target_pes) {
                auto path = topology.getPath(src_pe, dst_pe);
                if (path.size() < 2) { // 可能就是本地
                    pending[dst_pe].push_back({key, 0xFFFFFFFFu, {RouteDirection::LOCAL}, 11});
                    continue;
                }
                for (size_t i = 0; i + 1 < path.size(); ++i) {
                    children[path[i]].insert(path[i + 1]);
                }
                // 叶子：终点本地交付（LOCAL）
                pending[path.back()].push_back({key, 0xFFFFFFFFu, {RouteDirection::LOCAL}, 11});
            }
            // 为每个分叉/中间节点加入多出向方向
            for (const auto& kv : children) {
                PEId cur = kv.first;
                std::vector<RouteDirection> dirs;
                dirs.reserve(kv.second.size());
                for (auto nxt : kv.second) {
                    dirs.push_back(getRouteDirection(cur, nxt, topology));
                }
                if (!dirs.empty()) {
                    pending[cur].push_back({key, 0xFFFFFFFFu, dirs, 12});
                }
            }
        } else {
            // 单播：为每个目标沿最短路径添加出向与终点LOCAL
            for (auto t : targets) {
                PEId dst_pe = mapping.getNeuronPE(t);
                if (dst_pe == INVALID_PE_ID) continue;
                auto path = topology.getPath(src_pe, dst_pe);
                if (path.empty()) continue;
                for (size_t i = 0; i < path.size(); ++i) {
                    PEId cur = path[i];
                    if (i + 1 < path.size()) {
                        auto dir = getRouteDirection(cur, path[i + 1], topology);
                        pending[cur].push_back({key, 0xFFFFFFFFu, {dir}, 8});
                    } else {
                        pending[cur].push_back({key, 0xFFFFFFFFu, {RouteDirection::LOCAL}, 9});
                    }
                }
            }
        }
    }

    // 3) 去重：按 (key,mask) 合并路由方向
    struct VecHash { size_t operator()(const std::vector<RouteDirection>& v) const { size_t h=0; for(auto d:v) h = h*131 + static_cast<size_t>(d); return h; } };
    for (auto& [pe, vec] : pending) {
        // map (key,mask) -> routes set
        struct KeyMask { uint32_t k, m; bool operator==(const KeyMask& o) const { return k==o.k && m==o.m; } };
        struct KMHash { size_t operator()(const KeyMask& x) const { return (static_cast<size_t>(x.k) << 1) ^ x.m; } };

        std::unordered_map<KeyMask, std::vector<RouteDirection>, KMHash> km2routes;
        std::unordered_map<KeyMask, uint16_t, KMHash> km2prio;
        for (const auto& e : vec) {
            KeyMask km{e.key, e.mask};
            auto& routes = km2routes[km];
            for (auto d : e.routes) {
                if (std::find(routes.begin(), routes.end(), d) == routes.end()) routes.push_back(d);
            }
            auto& pr = km2prio[km];
            pr = std::max<uint16_t>(pr, e.priority);
        }
        // 回写
        vec.clear();
        vec.reserve(km2routes.size());
        for (auto& kv2 : km2routes) {
            vec.push_back({kv2.first.k, kv2.first.m, kv2.second, km2prio[kv2.first]});
        }
    }

    // 4) 前缀聚合（简化：仅尝试按最低位逐级合并；routes 完全一致方可合并）
    auto normalize_routes = [](const std::vector<RouteDirection>& v){
        std::vector<RouteDirection> r = v; std::sort(r.begin(), r.end()); r.erase(std::unique(r.begin(), r.end()), r.end()); return r; };

    auto try_prefix_aggregate = [&](std::vector<EntrySpec>& entries) {
        // 仅进行“LSB 连续位”聚合：mask 形如 0xFFFFFFFF << t，逐步提升 t
        // 从 t=0 到 15，迭代合并相邻键（低 t 位相同，且第 t 位 0/1 各一条），routes 完全一致
        size_t total_merges = 0;
        // 先规范化 routes，便于比较
        for (auto& e : entries) e.routes = normalize_routes(e.routes);

        for (uint32_t t = 0; t < 16; ++t) {
            const uint32_t required_mask = 0xFFFFFFFFu << t; // 低 t 位为0
            const uint32_t next_mask = 0xFFFFFFFFu << (t + 1);
            const uint32_t low_t_mask = (t == 0) ? 0u : ((1u << t) - 1u);
            const uint32_t bit_t = (1u << t);

            struct KeyRoutes {
                uint32_t base; // 清零低 t+1 位后的基地址
                std::vector<RouteDirection> routes;
                bool operator==(const KeyRoutes& o) const { return base == o.base && routes == o.routes; }
            };
            struct KRHash { size_t operator()(KeyRoutes const& kr) const { size_t h = kr.base; for (auto d: kr.routes) h = h * 131 + static_cast<size_t>(d); return h; } };

            // 记录候选对：key_lowbit = 0 和 1
            struct PairIdx { int idx0 = -1; int idx1 = -1; };
            std::unordered_map<KeyRoutes, PairIdx, KRHash> table;

            // 索引满足当前 t 的条目
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                if (e.mask != required_mask) continue;
                // 要求低 t 位为 0，避免非对齐合并
                if ((e.key & low_t_mask) != 0) continue;
                KeyRoutes kr{ e.key & ~((1u << (t + 1)) - 1u), e.routes };
                bool is_one = (e.key & bit_t) != 0;
                auto& pr = table[kr];
                if (is_one) { if (pr.idx1 == -1) pr.idx1 = static_cast<int>(i); }
                else { if (pr.idx0 == -1) pr.idx0 = static_cast<int>(i); }
            }

            // 标记要合并的对
            std::vector<char> used(entries.size(), 0);
            std::vector<EntrySpec> additions;
            for (const auto& kv : table) {
                const auto& pr = kv.second;
                if (pr.idx0 != -1 && pr.idx1 != -1) {
                    const auto& e0 = entries[pr.idx0];
                    const auto& e1 = entries[pr.idx1];
                    // routes 已按 normalize 比较相等
                    EntrySpec ne;
                    ne.key = kv.first.base; // 对齐到 2^(t+1)
                    ne.mask = next_mask;    // 低 t+1 位为 don't-care
                    ne.routes = e0.routes;
                    ne.priority = std::max(e0.priority, e1.priority);
                    additions.push_back(ne);
                    used[pr.idx0] = used[pr.idx1] = 1;
                    total_merges++;
                }
            }

            if (!additions.empty()) {
                std::vector<EntrySpec> kept;
                kept.reserve(entries.size());
                for (size_t i = 0; i < entries.size(); ++i) if (!used[i]) kept.push_back(entries[i]);
                kept.insert(kept.end(), additions.begin(), additions.end());
                entries.swap(kept);
                // 继续下一位 t+1 在同一 while外层循环
            }
        }
        return total_merges;
    };

    size_t before_entries = 0;
    for (const auto& [pe, vec] : pending) before_entries += vec.size();

    if (config_.enable_compression && config_.compression == CompressionMethod::PREFIX_AGGREGATION) {
        for (auto& [pe, vec] : pending) { (void)try_prefix_aggregate(vec); }
    }

    // 5) 构造路由表
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> tables;
    for (auto pe_id : topology.getAllPEIds()) {
        auto table = std::make_unique<RoutingTable>(pe_id);
        auto it = pending.find(pe_id);
        if (it != pending.end()) {
            for (const auto& e : it->second) {
                if (e.routes.size() == 1) table->addUnicastRoute(e.key, e.mask, e.routes[0], e.priority);
                else table->addMulticastRoute(e.key, e.mask, e.routes, e.priority);
            }
        }
        tables.emplace(pe_id, std::move(table));
    }

    // 6) 更新统计
    stats_.reset();
    stats_.total_neurons = network.getNeuronCount();
    stats_.total_connections = network.getConnectionCount();
    size_t after_entries = 0; for (const auto& [pe, tbl] : tables) after_entries += tbl->getTableSize();
    stats_.routing_table_entries = after_entries;
    // 估计平均路径长度
    if (stats_.total_connections > 0) {
        double total_hops = 0.0; uint32_t cnt = 0;
        for (const auto& c : network.getAllConnections()) {
            PEId sp = mapping.getNeuronPE(c.source_id), dp = mapping.getNeuronPE(c.target_id);
            if (sp == INVALID_PE_ID || dp == INVALID_PE_ID) continue;
            auto path = topology.getPath(sp, dp);
            if (path.size() >= 2) { total_hops += static_cast<int>(path.size() - 1); cnt++; }
        }
        if (cnt > 0) stats_.average_path_length = static_cast<float>(total_hops / cnt);
    }
    // 估计多播组数（fanout>=3）
    std::unordered_map<NeuronId, uint32_t> fan;
    for (const auto& c : network.getAllConnections()) fan[c.source_id]++;
    for (const auto& kv : fan) if (kv.second >= 3) stats_.multicast_groups++;
    // 压缩比
    stats_.compression_ratio = (before_entries > 0) ? static_cast<float>(after_entries) / static_cast<float>(before_entries) : 1.0f;

    return tables;
}

std::unordered_map<PEId, std::unique_ptr<RoutingTable>> RoutingTableGenerator::generateShortestPathRouting(
    const NeuralNetwork& network, const HardwareTopology& topology, const MappingSolution& mapping) {
    // 调用通用入口以生成最小可用路由表
    return generateRoutingTables(network, topology, mapping, config_);
}

std::unordered_map<PEId, std::unique_ptr<RoutingTable>> RoutingTableGenerator::generateLoadBalancedRouting(
    const NeuralNetwork& network, const HardwareTopology& topology, const MappingSolution& mapping, float) {
    return generateRoutingTables(network, topology, mapping, config_);
}

std::unordered_map<PEId, std::unique_ptr<RoutingTable>> RoutingTableGenerator::generateMulticastOptimizedRouting(
    const NeuralNetwork& network, const HardwareTopology& topology, const MappingSolution& mapping) {
    return generateRoutingTables(network, topology, mapping, config_);
}

std::unordered_map<PEId, std::unique_ptr<RoutingTable>> RoutingTableGenerator::generateFaultTolerantRouting(
    const NeuralNetwork& network, const HardwareTopology& topology, const MappingSolution& mapping, uint8_t) {
    return generateRoutingTables(network, topology, mapping, config_);
}

std::unordered_map<std::string, float> RoutingTableGenerator::compressRoutingTables(
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
    CompressionMethod method, float /*target_ratio*/) {
    size_t before = 0, after = 0;
    for (const auto& [pe, tbl] : routing_tables) before += tbl->getTableSize();

    if (method == CompressionMethod::PREFIX_AGGREGATION) {
        for (auto& [pe, tbl] : routing_tables) {
            performPrefixAggregation(*tbl);
        }
    }

    for (const auto& [pe, tbl] : routing_tables) after += tbl->getTableSize();
    float ratio = (before > 0) ? static_cast<float>(after) / static_cast<float>(before) : 1.0f;
    return {{"compression_ratio", ratio}};
}

// 简化版前缀聚合：仅在掩码全1、routes完全相同的条目间做两两合并（按低位逐步聚合）
size_t RoutingTableGenerator::performPrefixAggregation(RoutingTable& table) {
    // 拿到可修改视图：通过导出/导入简化访问（避免暴露内部）
    // 为降低复杂度，我们直接访问私有成员不可行，改为复制-重建：
    // 这里采用反射式重建（获取导出JSON再解析会复杂），因此选择在本文件范围内做近似：

    // 方案：利用 findAllMatches/lookupRoute 无法遍历条目；因此在当前设计下，
    // 我们保守地返回0（若缺少条目枚举API）。
    // 为确保阶段一可用，再提供小型合并：对同一键的重复ENTRY进行去重。

    // NOTE: 由于RoutingTable未提供遍历接口（仅getTableSize/getStatistics等），
    // 在不修改头文件的前提下无法安全访问全部entries_。
    // 暂以0返回，并依靠addEntry前的策略减少重复；后续若允许扩展接口再完善。
    return 0;
}
size_t RoutingTableGenerator::performRouteMerging(RoutingTable&) { return 0; }
size_t RoutingTableGenerator::performMulticastTreeCompression(
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>>&, const HardwareTopology&) { return 0; }

std::vector<MulticastGroup> RoutingTableGenerator::detectMulticastGroups(
    const NeuralNetwork& network, const MappingSolution& mapping, size_t min_group_size) {
    std::vector<MulticastGroup> groups;
    // 按源神经元聚合扇出
    std::unordered_map<NeuronId, std::vector<NeuronId>> fanouts;
    for (const auto& c : network.getAllConnections()) fanouts[c.source_id].push_back(c.target_id);

    uint16_t next_gid = 1;
    for (const auto& [src, tgts] : fanouts) {
        if (tgts.size() < min_group_size) continue;
        MulticastGroup g;
        g.group_id = next_gid++;
        // 成员：目标神经元的全局ID（用于统计/导出）
        for (auto t : tgts) {
            PEId tpe = mapping.getNeuronPE(t);
            if (tpe == INVALID_PE_ID) continue;
            g.members.push_back(AddressEvent::generateGlobalId(t, tpe, 0));
            g.target_pes.push_back(tpe);
        }
        std::sort(g.target_pes.begin(), g.target_pes.end());
        g.target_pes.erase(std::unique(g.target_pes.begin(), g.target_pes.end()), g.target_pes.end());
        groups.push_back(std::move(g));
    }
    return groups;
}

std::unordered_map<PEId, std::vector<RouteDirection>> RoutingTableGenerator::buildMulticastTree(
    const MulticastGroup& group, const HardwareTopology& topology, PEId root_pe) {
    std::unordered_map<PEId, std::vector<RouteDirection>> outdirs;
    if (group.target_pes.empty()) return outdirs;
    PEId root = root_pe;
    if (root == INVALID_PE_ID) {
        // 选取到各目标距离和最小的PE为根（近似），退化为第一个目标
        float best_sum = std::numeric_limits<float>::infinity();
        for (auto candidate : group.target_pes) {
            float sum = 0.0f;
            for (auto t : group.target_pes) {
                int32_t d = topology.getDistance(candidate, t);
                sum += (d >= 0) ? static_cast<float>(d) : 1e6f;
            }
            if (sum < best_sum) { best_sum = sum; root = candidate; }
        }
        if (root == INVALID_PE_ID) root = group.target_pes.front();
    }
    // 并集最短路径形成树
    std::unordered_map<PEId, std::unordered_set<PEId>> children;
    for (auto t : group.target_pes) {
        auto path = topology.getPath(root, t);
        for (size_t i = 0; i + 1 < path.size(); ++i) children[path[i]].insert(path[i + 1]);
    }
    // 映射为方向列表
    for (const auto& kv : children) {
        PEId u = kv.first;
        for (auto v : kv.second) {
            outdirs[u].push_back(getRouteDirection(u, v, topology));
        }
    }
    // 叶子无需出向（仅 LOCAL），由调用侧处理
    return outdirs;
}

std::unordered_map<std::string, float> RoutingTableGenerator::optimizeMulticastTrees(
    std::unordered_map<uint16_t, std::unordered_map<PEId, std::vector<RouteDirection>>>&, const HardwareTopology&) { return {}; }

std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> RoutingTableGenerator::assignGlobalNeuronIds(
    const NeuralNetwork&, const MappingSolution&) { return {}; }

std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> RoutingTableGenerator::generateNeuronIdMapping(
    const NeuralNetwork&, const MappingSolution&, PEId) { return {}; }

std::vector<std::string> RoutingTableGenerator::validateGlobalIdAssignment(
    const std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId>&, const NeuralNetwork&) { return {}; }

std::vector<PEId> RoutingTableGenerator::calculateNeuronPath(
    NeuronId, NeuronId, const MappingSolution&, const HardwareTopology&) { return {}; }

std::unordered_map<uint32_t, uint32_t> RoutingTableGenerator::analyzePathLengthDistribution(
    const NeuralNetwork&, const MappingSolution&, const HardwareTopology&) { return {}; }

std::vector<std::pair<PEId, float>> RoutingTableGenerator::identifyRoutingBottlenecks(
    const NeuralNetwork&, const MappingSolution&, const HardwareTopology&) { return {}; }

std::string RoutingTableGenerator::generateRoutingReport(
    const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
    const std::string& format) {
    // 聚合统计
    size_t total_entries = 0;
    for (const auto& [pe, tbl] : routing_tables) total_entries += tbl->getTableSize();

    // 统计每PE表项
    size_t max_entries = 0; double sum_entries = 0.0;
    for (const auto& [pe, tbl] : routing_tables) { size_t n = tbl->getTableSize(); max_entries = std::max(max_entries, n); sum_entries += n; }

    if (format == "text") {
        std::ostringstream oss;
        oss << "Routing Generation Report\n";
        oss << "- total_pes: " << routing_tables.size() << "\n";
        oss << "- total_entries: " << total_entries << "\n";
        oss << "- avg_path_len: " << stats_.average_path_length << "\n";
        oss << "- multicast_groups: " << stats_.multicast_groups << "\n";
        oss << "- compression_ratio: " << stats_.compression_ratio << "\n";
        oss << "- avg_entries_per_pe: " << (routing_tables.empty()?0.0:sum_entries / routing_tables.size()) << "\n";
        oss << "- max_entries_per_pe: " << max_entries << "\n";
        return oss.str();
    }

    // json
    std::ostringstream oss;
    oss << "{\"total_pes\":" << routing_tables.size()
        << ",\"total_entries\":" << total_entries
        << ",\"average_path_length\":" << stats_.average_path_length
        << ",\"multicast_groups\":" << stats_.multicast_groups
        << ",\"compression_ratio\":" << stats_.compression_ratio
        << ",\"avg_entries_per_pe\":" << (routing_tables.empty()?0.0:sum_entries / routing_tables.size())
        << ",\"max_entries_per_pe\":" << max_entries
        << "}";
    return oss.str();
}

std::string RoutingTableGenerator::exportVisualizationData(
    const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
    const HardwareTopology& /*topology*/, const std::string& format) {
    if (format == "csv") {
        std::ostringstream oss;
        oss << "pe_id,entries\n";
        for (const auto& [pe, tbl] : routing_tables) {
            oss << pe << "," << tbl->getTableSize() << "\n";
        }
        return oss.str();
    }

    // 默认json：每个PE输出其路由表JSON（嵌套对象）
    std::ostringstream oss;
    oss << "{";
    size_t i = 0, n = routing_tables.size();
    for (const auto& [pe, tbl] : routing_tables) {
        oss << "\"pe_" << pe << "\":" << tbl->exportTable("json");
        if (++i < n) oss << ",";
    }
    oss << "}";
    return oss.str();
}

std::string RoutingTableGenerator::debugConnectionRouting(
    NeuronId, NeuronId, const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>&, const MappingSolution&, const HardwareTopology&) { return {}; }

void RoutingTableGenerator::updateStatistics(const std::string&, float) {}
RouteDirection RoutingTableGenerator::getRouteDirection(PEId from_pe, PEId to_pe, const HardwareTopology& topology) {
    if (from_pe == to_pe) return RouteDirection::LOCAL;

    const std::string topo = topology.getTopologyType();
    int64_t diff = static_cast<int64_t>(to_pe) - static_cast<int64_t>(from_pe);

    if (topo == "mesh2d" || topo == "torus2d") {
        // 推断列宽cols：取 from_pe 邻居中最小的 >1 的差值
        auto neighbors = topology.getNeighbors(from_pe);
        uint32_t cols = 0;
        for (auto n : neighbors) {
            uint64_t ad = (from_pe > n) ? (from_pe - n) : (n - from_pe);
            if (ad > 1 && (cols == 0 || ad < cols)) cols = static_cast<uint32_t>(ad);
        }
        if (diff == 1) return RouteDirection::EAST;
        if (diff == -1) return RouteDirection::WEST;
        if (cols != 0) {
            if (diff == static_cast<int64_t>(cols)) return RouteDirection::SOUTH;
            if (diff == -static_cast<int64_t>(cols)) return RouteDirection::NORTH;
        }
        // 简单处理环回边：大正差视作北/西，小负差视作南/东
        if (diff > 1) return RouteDirection::SOUTH;
        if (diff < -1) return RouteDirection::NORTH;
        return RouteDirection::EAST;
    }

    if (topo == "mesh3d") {
        // 从 from_pe 邻居差值中获取三个基差：1、z_dim、y_dim*z_dim
        auto neighbors = topology.getNeighbors(from_pe);
        uint32_t a = 0, b = 0; // a=min>1, b=次小>1
        for (auto n : neighbors) {
            uint64_t ad = (from_pe > n) ? (from_pe - n) : (n - from_pe);
            if (ad == 1) continue;
            if (a == 0 || ad < a) { b = a; a = static_cast<uint32_t>(ad); }
            else if (b == 0 || ad < b) { b = static_cast<uint32_t>(ad); }
        }
        // 1 -> Z轴(E/W), a-> Y轴(N/S), b-> X轴(UP/DOWN)
        if (diff == 1) return RouteDirection::EAST;
        if (diff == -1) return RouteDirection::WEST;
        if (a) {
            if (diff == static_cast<int64_t>(a)) return RouteDirection::SOUTH;
            if (diff == -static_cast<int64_t>(a)) return RouteDirection::NORTH;
        }
        if (b) {
            if (diff == static_cast<int64_t>(b)) return RouteDirection::UP;
            if (diff == -static_cast<int64_t>(b)) return RouteDirection::DOWN;
        }
        return RouteDirection::EAST;
    }

    // 其他拓扑：使用通用默认值
    return RouteDirection::EAST;
}

std::vector<PEId> RoutingTableGenerator::findShortestPath(PEId source, PEId target, const HardwareTopology& topology) {
    return topology.getPath(source, target);
}
std::vector<std::vector<PEId>> RoutingTableGenerator::findKShortestPaths(PEId, PEId, const HardwareTopology&, uint8_t) { return {}; }
void RoutingTableGenerator::generateUnicastRoutes(RoutingTable&, const NeuralNetwork&, const MappingSolution&, const HardwareTopology&, PEId) {}
void RoutingTableGenerator::generateMulticastRoutes(RoutingTable&, const std::vector<MulticastGroup>&, const HardwareTopology&, PEId) {}
bool RoutingTableGenerator::canMergeRoutingEntries(const RoutingEntry&, const RoutingEntry&) { return false; }
RoutingEntry RoutingTableGenerator::mergeRoutingEntries(const RoutingEntry& e1, const RoutingEntry&) { return e1; }
uint32_t RoutingTableGenerator::findCommonPrefix(uint32_t, uint32_t, uint32_t, uint32_t) { return 0; }
PEId RoutingTableGenerator::findOptimalMulticastRoot(const MulticastGroup&, const HardwareTopology&) { return INVALID_PE_ID; }
std::vector<PEId> RoutingTableGenerator::computeMulticastSpanningTree(const std::vector<PEId>&, PEId, const HardwareTopology&) { return {}; }
float RoutingTableGenerator::calculatePELoad(PEId, const NeuralNetwork&, const MappingSolution&) { return 0.0f; }
std::vector<PEId> RoutingTableGenerator::selectAlternativePaths(PEId, PEId, const HardwareTopology&, uint8_t) { return {}; }
bool RoutingTableGenerator::validateRouteConnectivity(const RoutingTable&, const HardwareTopology&) { return true; }
void RoutingTableGenerator::logRoutingDecision(const std::string&, const std::vector<std::string>&) {}
const std::vector<PEId>* RoutingTableGenerator::getCachedPath(PEId, PEId) { return nullptr; }
void RoutingTableGenerator::cachePath(PEId, PEId, const std::vector<PEId>&) {}

// === 导出公共API实现 ===
bool RoutingTableGenerator::exportArtifacts(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingSolution& mapping,
    const std::string& out_dir,
    const RoutingGenerationConfig& gen_cfg) {
    // 生成路由表
    auto tables = generateRoutingTables(network, topology, mapping, gen_cfg);

    // 创建目录
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    // 导出 mapping.json
    std::ostringstream mjson;
    mjson << "{\"assignments\":[";
    auto assigns = mapping.getAllAssignments();
    for (size_t i = 0; i < assigns.size(); ++i) {
        const auto& a = assigns[i];
        mjson << "{\"neuron_id\":" << a.neuron_id
              << ",\"pe_id\":" << a.pe_id
              << ",\"core_id\":" << a.core_id << "}";
        if (i + 1 < assigns.size()) mjson << ",";
    }
    mjson << "]}";
    std::ofstream(out_dir + "/mapping.json") << mjson.str();

    // 导出路由表（JSON）
    std::string routing_json = exportVisualizationData(tables, topology, "json");
    std::ofstream(out_dir + "/routing_tables.json") << routing_json;

    // 导出统计CSV
    std::string csv = exportVisualizationData(tables, topology, "csv");
    std::ofstream(out_dir + "/routing_stats.csv") << csv;

    // 导出报告
    std::string report = generateRoutingReport(tables, "json");
    std::ofstream(out_dir + "/report.json") << report;

    return true;
}

// ========================= MulticastGroup ========================
// 为满足链接期需要，此处不实现 MulticastGroup/Manager 复杂逻辑；
// 头文件主要为接口声明，examples 不在 test-compile 目标内。

} // namespace NeuronMapping

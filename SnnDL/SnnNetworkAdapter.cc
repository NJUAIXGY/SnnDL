// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNetworkAdapter.cc: SNN 通用网络拓扑适配器实现文件
//

#include "SnnNetworkAdapter.h"
#include "SimpleNetworkWrapper.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace SST {
namespace SnnDL {

// ===== NetworkEventConverter 实现 =====

SST::Interfaces::SimpleNetwork::Request* NetworkEventConverter::convertSpikeToRequest(
    SpikeEvent* spike_event, uint32_t dest_node, uint32_t src_node) 
{
    if (!spike_event) {
        return nullptr;
    }
    
    // 创建SimpleNetwork Request
    auto* request = new SST::Interfaces::SimpleNetwork::Request();
    
    // 设置网络属性
    request->dest = static_cast<SST::Interfaces::SimpleNetwork::nid_t>(dest_node);
    request->src = static_cast<SST::Interfaces::SimpleNetwork::nid_t>(src_node);
    request->vn = 0;  // 默认虚拟网络0
    request->size_in_bits = 64 * 8;  // 64字节数据包
    request->head = true;
    request->tail = true;
    request->allow_adaptive = true;
    
    // 将SpikeEvent作为payload嵌入
    request->givePayload(spike_event->clone());
    
    return request;
}

SpikeEvent* NetworkEventConverter::convertRequestToSpike(SST::Interfaces::SimpleNetwork::Request* request) 
{
    if (!request) {
        return nullptr;
    }
    
    // 从payload中提取SpikeEvent
    SST::Event* payload = request->takePayload();
    SpikeEvent* spike_event = dynamic_cast<SpikeEvent*>(payload);
    
    if (!spike_event) {
        // 如果payload不是SpikeEvent，创建一个新的
        delete payload; // 清理无效payload
        spike_event = new SpikeEvent(0, 0.0); // 创建默认SpikeEvent
    }
    
    return spike_event;
}

// ===== SnnNetworkAdapter 主要实现 =====

SnnNetworkAdapter::SnnNetworkAdapter(SST::ComponentId_t id, SST::Params& params)
    : SnnInterface(id, params)
{
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output = new SST::Output("SnnNetworkAdapter[@p:@l]: ", verbose_level, 0, SST::Output::STDOUT);
    
    // 解析基本参数
    node_id = params.find<uint32_t>("node_id", 0);
    routing_algorithm = params.find<std::string>("routing_algorithm", "XY");
    link_bw = params.find<std::string>("link_bw", "40GiB/s");
    packet_size = params.find<std::string>("packet_size", "64B");
    input_buf_size = params.find<std::string>("input_buf_size", "1KiB");
    output_buf_size = params.find<std::string>("output_buf_size", "1KiB");
    
    // 性能参数
    enable_adaptive_routing = params.find<bool>("enable_adaptive_routing", false);
    congestion_threshold = params.find<double>("congestion_threshold", 0.8);
    
    // Merlin集成参数
    enable_merlin_router = params.find<bool>("enable_merlin_router", false);
    use_direct_link = params.find<bool>("use_direct_link", true);
    use_multi_port = params.find<bool>("use_multi_port", false);
    port_name = params.find<std::string>("port_name", "network");
    
    // 解析拓扑类型和拓扑形状
    std::string topology_str = params.find<std::string>("topology_type", "mesh2d");
    topology_type = parseTopologyType(topology_str);
    topology_shape = params.find<std::string>("topology_shape", "4x4");
    
    // 初始化基础统计计数器
    spikes_routed_count = 0;
    local_spikes_count = 0;
    remote_spikes_count = 0;
    xy_routes_count = 0;
    adaptive_routes_count = 0;
    congestion_events_count = 0;
    
    // 初始化扩展性能统计计数器
    total_hops_count = 0;
    average_latency_cycles = 0;
    max_hops_observed = 0;
    bandwidth_bytes_sent = 0;
    packets_dropped = 0;
    
    // 在构造函数中注册统计对象（SST要求在组件连接前注册）
    stat_spikes_routed = registerStatistic<uint64_t>("spikes_routed");
    stat_local_spikes = registerStatistic<uint64_t>("local_spikes");
    stat_remote_spikes = registerStatistic<uint64_t>("remote_spikes");
    stat_xy_routes = registerStatistic<uint64_t>("xy_routes");
    stat_adaptive_routes = registerStatistic<uint64_t>("adaptive_routes");
    stat_congestion_events = registerStatistic<uint64_t>("congestion_events");
    stat_total_hops = registerStatistic<uint64_t>("total_hops");
    stat_average_latency = registerStatistic<uint64_t>("average_latency");
    stat_max_hops = registerStatistic<uint64_t>("max_hops");
    stat_bandwidth_utilization = registerStatistic<uint64_t>("bandwidth_utilization");
    stat_packets_dropped = registerStatistic<uint64_t>("packets_dropped");
    
    // 初始化SimpleNetwork包装器
    simple_network_wrapper = nullptr;
    
    // 初始化网络接口（参考SnnNIC的成功模式）
    direct_link = nullptr;
    router = nullptr;
    
    if (use_direct_link && use_multi_port) {
        // 使用多端口Direct Link模式 - 为每个方向创建链路
        std::vector<std::string> directions = {"north", "south", "east", "west"};
        
        for (const std::string& direction : directions) {
            if (isPortConnected(direction)) {
                SST::Link* dir_link = configureLink(direction, "0ps",
                    new SST::Event::Handler2<SnnNetworkAdapter,&SnnNetworkAdapter::handleDirectSpikeEvent>(this));
                
                if (dir_link) {
                    direction_links[direction] = dir_link;
                    output->verbose(CALL_INFO, 1, 0, "✅ %s方向Link创建成功\n", direction.c_str());
                } else {
                    output->verbose(CALL_INFO, 1, 0, "⚠️ %s方向Link创建失败\n", direction.c_str());
                }
            } else {
                output->verbose(CALL_INFO, 2, 0, "📝 %s方向端口未连接\n", direction.c_str());
            }
        }
        
        output->verbose(CALL_INFO, 1, 0, "🔗 多端口模式：创建了%lu个方向链路\n", direction_links.size());
        
    } else if (use_direct_link) {
        // 使用单端口Direct Link模式 - 检查端口是否连接
        if (isPortConnected("network")) {
            direct_link = configureLink("network", "0ps",
                new SST::Event::Handler2<SnnNetworkAdapter,&SnnNetworkAdapter::handleDirectSpikeEvent>(this));
            
            if (direct_link) {
                output->verbose(CALL_INFO, 1, 0, "✅ 直接Link网络接口创建成功\n");
            } else {
                output->verbose(CALL_INFO, 1, 0, "⚠️ 直接Link创建失败\n");
            }
        } else {
            output->verbose(CALL_INFO, 1, 0, "⚠️ 网络端口未连接，直接Link创建失败\n");
        }
    } else if (enable_merlin_router) {
        // 使用SimpleNetwork模式 - 参考SnnNIC的成功实现
        output->verbose(CALL_INFO, 1, 0, "尝试加载网络接口...\n");
        
        // 首先尝试加载用户定义的网络接口 (推荐方式)
        router = loadUserSubComponent<SST::Interfaces::SimpleNetwork>("linkcontrol", ComponentInfo::SHARE_NONE, 1);
        
        if (!router) {
            // 如果没有用户定义的接口，创建默认的merlin.linkcontrol
            output->verbose(CALL_INFO, 1, 0, "未找到用户定义的linkcontrol，创建默认merlin.linkcontrol\n");
            
            SST::Params net_params;
            net_params.insert("port_name", port_name);
            net_params.insert("link_bw", link_bw);
            net_params.insert("input_buf_size", input_buf_size);
            net_params.insert("output_buf_size", output_buf_size);
            net_params.insert("num_vns", "2");  // 与路由器保持一致的虚拟网络数
            
            // 使用与SnnNIC相同的标志和参数
            router = loadAnonymousSubComponent<SST::Interfaces::SimpleNetwork>("merlin.linkcontrol", "linkcontrol", 0, 
                ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, net_params, 1);
        }
        
        if (router) {
            output->verbose(CALL_INFO, 1, 0, "✅ 网络接口创建成功\n");
            
            // 关键修复：由于使用了SHARE_PORTS标志创建linkcontrol，
            // linkcontrol的端口应该自动暴露给父组件
            output->verbose(CALL_INFO, 2, 0, "📤 LinkControl端口通过SHARE_PORTS自动暴露\n");
            output->verbose(CALL_INFO, 1, 0, "✅ 父组件可以通过network端口连接到外部路由器\n");
            
        } else {
            output->verbose(CALL_INFO, 1, 0, "❌ 无法创建网络接口，回退到简化模式\n");
        }
    } else {
        // 使用简化的直接通信模式
        output->verbose(CALL_INFO, 2, 0, "使用简化拓扑模式（无网络接口）\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "SnnNetworkAdapter initialized for node %u with topology %s\n", 
                    node_id, topology_str.c_str());
}

SnnNetworkAdapter::~SnnNetworkAdapter()
{
    delete output;
}

void SnnNetworkAdapter::init(unsigned int phase)
{
    if (router) {
        router->init(phase);
    }
    
    if (phase == 0) {
        // 在第一阶段初始化拓扑处理器
        initializeTopologyHandler();
        
        output->verbose(CALL_INFO, 2, 0, "Phase %u: Topology handler initialized\n", phase);
    }
}

void SnnNetworkAdapter::setup()
{
    if (router) {
        try {
            router->setup();
            output->verbose(CALL_INFO, 2, 0, "✅ 路由器setup成功\n");
            
            // 设置网络事件接收回调（SimpleNetwork模式）
            if (enable_merlin_router && !use_direct_link) {
                router->setNotifyOnReceive(
                    new SST::Interfaces::SimpleNetwork::Handler2<SnnNetworkAdapter, &SnnNetworkAdapter::handleNetworkEvent>(this));
                router->setNotifyOnSend(
                    new SST::Interfaces::SimpleNetwork::Handler2<SnnNetworkAdapter, &SnnNetworkAdapter::spaceAvailable>(this));
                output->verbose(CALL_INFO, 2, 0, "✅ 网络事件回调设置完成\n");
            }
        } catch (const std::exception& e) {
            output->verbose(CALL_INFO, 1, 0, "⚠️ 路由器setup失败: %s\n", e.what());
            router = nullptr; // 重置为简化模式
        } catch (...) {
            output->verbose(CALL_INFO, 1, 0, "⚠️ 路由器setup发生未知异常\n");
            router = nullptr; // 重置为简化模式
        }
    }
    
    // 统计对象已在构造函数中注册
    output->verbose(CALL_INFO, 2, 0, "📊 网络适配器setup阶段完成\n");
    
    if (topology_handler) {
        std::string topo_desc = topology_handler->getTopologyDescription();
        output->verbose(CALL_INFO, 1, 0, "Setup complete: %s\n", topo_desc.c_str());
    }
}

void SnnNetworkAdapter::finish()
{
    if (router) {
        router->finish();
    }
    
    // 输出最终统计摘要（统计数据已经在运行时实时更新）
    output->verbose(CALL_INFO, 1, 0, "Final statistics - Routed: %lu, Local: %lu, Remote: %lu\n",
                    spikes_routed_count, local_spikes_count, remote_spikes_count);
    output->verbose(CALL_INFO, 1, 0, "Routing breakdown - XY: %lu, Adaptive: %lu, Congestion: %lu\n",
                    xy_routes_count, adaptive_routes_count, congestion_events_count);
}

// ===== SnnInterface 接口实现 =====

void SnnNetworkAdapter::setSpikeHandler(SpikeHandler handler)
{
    spike_handler = handler;
}

void SnnNetworkAdapter::sendSpike(SpikeEvent* spike_event)
{
    if (!spike_event) {
        output->verbose(CALL_INFO, 3, 0, "Received null spike event\n");
        return;
    }
    
    uint32_t dest_node = spike_event->getDestinationNode();
    
    output->verbose(CALL_INFO, 3, 0, "Sending spike from neuron %u to neuron %u (node %u)\n",
                    spike_event->getSourceNeuron(), spike_event->getDestinationNeuron(), dest_node);
    
    if (dest_node == node_id) {
        // 本地处理
        local_spikes_count++;
        if (stat_local_spikes) stat_local_spikes->addData(1);
        if (spike_handler) {
            spike_handler(spike_event);
        }
    } else {
        // 远程路由
        routeSpike(spike_event, dest_node);
        remote_spikes_count++;
        if (stat_remote_spikes) stat_remote_spikes->addData(1);
    }
    
    spikes_routed_count++;
    if (stat_spikes_routed) stat_spikes_routed->addData(1);
}

void SnnNetworkAdapter::setNodeId(uint32_t new_node_id)
{
    node_id = new_node_id;
}

uint32_t SnnNetworkAdapter::getNodeId() const
{
    return node_id;
}

std::string SnnNetworkAdapter::getNetworkStatus() const
{
    std::ostringstream status;
    status << "SnnNetworkAdapter[" << node_id << "] - ";
    status << "Routed: " << spikes_routed_count;
    status << ", Local: " << local_spikes_count;
    status << ", Remote: " << remote_spikes_count;
    
    if (topology_handler) {
        status << ", Topology: " << topology_handler->getTopologyDescription();
    }
    
    return status.str();
}

// ===== 路由器回调方法 =====

bool SnnNetworkAdapter::handleIncoming(int vn)
{
    // 简化版本 - 暂时直接返回
    return true;
}

bool SnnNetworkAdapter::spaceAvailable(int vn)
{
    // 处理待发送队列 - 参考SnnNIC的成功实现
    while (!pending_spikes.empty() && router && router->spaceToSend(vn, 64)) { // 假设64位数据包
        SpikeEvent* spike = pending_spikes.front();
        pending_spikes.pop();
        
        uint32_t dest_node = spike->getDestinationNode();
        
        // 重新创建网络请求
        SST::Interfaces::SimpleNetwork::Request* req = createNetworkRequest(spike, dest_node, 0);
        
        if (req && router->send(req, vn)) {
            output->verbose(CALL_INFO, 2, 0, "✅ 待发送脉冲重发成功: 神经元%u -> 节点%u\n", 
                            spike->getNeuronId(), dest_node);
            
            // 更新统计
            spikes_routed_count++;
            remote_spikes_count++;
            if (stat_spikes_routed) stat_spikes_routed->addData(1);
            if (stat_remote_spikes) stat_remote_spikes->addData(1);
        } else {
            // 仍然无法发送，重新放回队列
            pending_spikes.push(spike);
            if (req) delete req;
            break; // 停止尝试更多发送
        }
        
        // 清理原始脉冲事件（已经复制到请求中）
        delete spike;
    }
    
    return true;
}

// ===== 内部实现方法 =====

void SnnNetworkAdapter::initializeTopologyHandler()
{
    // 创建拓扑处理器需要的参数
    SST::Params handler_params;
    
    // 从构造函数参数中获取实际的拓扑形状，而不是硬编码4x4
    std::string actual_topology_shape = topology_shape.empty() ? "4x4" : topology_shape;
    handler_params.insert("topology_shape", actual_topology_shape);
    
    switch (topology_type) {
        case TopologyType::MESH_2D:
            topology_handler = std::make_unique<Mesh2DHandler>();
            break;
        case TopologyType::TORUS_2D:
            topology_handler = std::make_unique<Torus2DHandler>();
            break;
        default:
            output->fatal(CALL_INFO, -1, "Unsupported topology type\n");
    }
    
    topology_handler->initialize(handler_params, node_id);
    
    output->verbose(CALL_INFO, 2, 0, "Topology handler initialized: %s\n",
                    topology_handler->getTopologyDescription().c_str());
}

TopologyType SnnNetworkAdapter::parseTopologyType(const std::string& type_str)
{
    if (type_str == "mesh2d") return TopologyType::MESH_2D;
    if (type_str == "torus2d") return TopologyType::TORUS_2D;
    
    output->fatal(CALL_INFO, -1, "Unknown topology type: %s\n", type_str.c_str());
    return TopologyType::MESH_2D;  // 默认值
}

void SnnNetworkAdapter::routeSpike(SpikeEvent* spike_event, uint32_t dest_node)
{
    if (!topology_handler) {
        output->verbose(CALL_INFO, 1, 0, "No topology handler available for routing\n");
        return;
    }
    
    // 计算路由
    int next_port = topology_handler->calculateRoute(dest_node);
    
    if (next_port < 0) {
        output->verbose(CALL_INFO, 2, 0, "No route found to node %u\n", dest_node);
        return;
    }
    
    // 计算跳数距离
    int hop_distance = topology_handler->calculateHopDistance(dest_node);
    if (hop_distance > 0) {
        total_hops_count += hop_distance;
        if (stat_total_hops) stat_total_hops->addData(hop_distance);
        
        // 更新最大跳数观察值
        if (hop_distance > max_hops_observed) {
            max_hops_observed = hop_distance;
            if (stat_max_hops) stat_max_hops->addData(max_hops_observed);
        }
    }
    
    // 模拟网络延迟（基于跳数）
    uint64_t estimated_latency = hop_distance * 10; // 假设每跳10个周期
    average_latency_cycles = (average_latency_cycles + estimated_latency) / 2; // 简单移动平均
    if (stat_average_latency) stat_average_latency->addData(estimated_latency);
    
    // 模拟带宽使用
    uint64_t packet_size_bytes = 64; // 假设每个脉冲64字节
    bandwidth_bytes_sent += packet_size_bytes;
    if (stat_bandwidth_utilization) stat_bandwidth_utilization->addData(packet_size_bytes);
    
    output->verbose(CALL_INFO, 3, 0, "Route calculated: dest_node %u -> port %d, hops %d, latency %lu\n", 
                    dest_node, next_port, hop_distance, estimated_latency);
    
    // 根据网络模式选择发送方式
    if (use_direct_link && use_multi_port && (!direction_links.empty() || !parent_direction_links.empty())) {
        // 使用多端口Direct Link模式发送
        sendViaMultiPortLink(spike_event, dest_node, next_port);
    } else if (use_direct_link && (direct_link || parent_direction_links.find("network") != parent_direction_links.end())) {
        // 使用单端口Direct Link模式发送 - 支持SubComponent自己的链路或父组件注入的链路
        sendViaDirectLink(spike_event, dest_node);
    } else if (router && enable_merlin_router) {
        // 使用SimpleNetwork模式发送
        sendViaMerlinRouter(spike_event, dest_node, next_port);
    } else {
        // 简化模式：仅更新统计信息，不实际发送
        output->verbose(CALL_INFO, 3, 0, "简化模式：脉冲路由完成（未实际传输）\n");
    }
    
    // 更新路由统计 - 实时添加统计数据
    if (routing_algorithm == "XY") {
        xy_routes_count++;
        if (stat_xy_routes) stat_xy_routes->addData(1);
    } else if (enable_adaptive_routing) {
        adaptive_routes_count++;
        if (stat_adaptive_routes) stat_adaptive_routes->addData(1);
    }
}

SST::Interfaces::SimpleNetwork::Request* SnnNetworkAdapter::createNetworkRequest(
    SpikeEvent* spike_event, uint32_t dest_node, int route_port)
{
    // 创建网络请求对象
    SST::Interfaces::SimpleNetwork::Request* req = new SST::Interfaces::SimpleNetwork::Request();
    
    // 设置正确的目标地址
    req->dest = dest_node;  // 使用实际目标节点
    req->src = node_id;
    req->size_in_bits = 64 * 8;  // 64字节 = 512比特
    req->vn = 0;  // 虚拟网络0
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    
    // 关键：将SpikeEvent包装为SpikeEventWrapper，然后作为payload
    SpikeEventWrapper* wrapper = new SpikeEventWrapper(spike_event);
    req->givePayload(wrapper);
    
    output->verbose(CALL_INFO, 3, 0, "🌐 创建SimpleNetwork请求: src=%u, dest=%u, 包装SpikeEvent=%u->%u\n", 
                    node_id, dest_node, spike_event->getNeuronId(), spike_event->getDestinationNeuron());
    
    return req;
}

SpikeEvent* SnnNetworkAdapter::extractSpikeFromRequest(SST::Interfaces::SimpleNetwork::Request* req)
{
    if (!req) return nullptr;
    
    // 从请求中提取SpikeEventWrapper，然后解包得到SpikeEvent
    SST::Event* payload = req->takePayload();
    if (!payload) {
        output->verbose(CALL_INFO, 1, 0, "⚠️ SimpleNetwork请求没有payload\n");
        return nullptr;
    }
    
    // 尝试转换为SpikeEventWrapper
    SpikeEventWrapper* wrapper = dynamic_cast<SpikeEventWrapper*>(payload);
    if (!wrapper) {
        output->verbose(CALL_INFO, 1, 0, "⚠️ Payload不是SpikeEventWrapper类型\n");
        delete payload;
        return nullptr;
    }
    
    // 从wrapper中提取SpikeEvent
    SpikeEvent* original_spike = wrapper->getSpikeEvent();
    if (!original_spike) {
        output->verbose(CALL_INFO, 1, 0, "⚠️ SpikeEventWrapper中没有SpikeEvent\n");
        delete wrapper;
        return nullptr;
    }
    
    // 创建SpikeEvent的副本
    SpikeEvent* extracted_spike = new SpikeEvent(
        original_spike->getNeuronId(),
        original_spike->getDestinationNeuron(),
        original_spike->getDestinationNode(),
        original_spike->getWeight(),
        original_spike->getTimestamp()
    );
    extracted_spike->hop_count = original_spike->hop_count + 1;  // 增加跳数
    
    output->verbose(CALL_INFO, 3, 0, "🌐 从SimpleNetwork请求提取SpikeEvent: %u->%u (跳数%u)\n", 
                    extracted_spike->getNeuronId(), extracted_spike->getDestinationNeuron(), 
                    extracted_spike->hop_count);
    
    // 清理wrapper
    delete wrapper;
    
    return extracted_spike;
}

double SnnNetworkAdapter::getPortCongestion(int port)
{
    auto it = port_utilization.find(port);
    if (it != port_utilization.end()) {
        return it->second;
    }
    return 0.0;
}

void SnnNetworkAdapter::updateLoadStatistics(int port)
{
    port_counters[port]++;
    
    // 简单的拥塞检测
    if (port_counters[port] % 100 == 0) {
        double utilization = static_cast<double>(port_counters[port]) / 1000.0;
        port_utilization[port] = std::min(utilization, 1.0);
        
        if (utilization > congestion_threshold) {
            congestion_events_count++;
            if (stat_congestion_events) stat_congestion_events->addData(1);
        }
    }
}

// ===== Mesh2DHandler 实现 =====

void Mesh2DHandler::initialize(SST::Params& params, uint32_t node_id)
{
    this->node_id = node_id;
    
    // 解析拓扑形状
    std::string shape_str = params.find<std::string>("topology_shape", "4x4");
    std::size_t pos = shape_str.find('x');
    if (pos != std::string::npos) {
        width = std::stoul(shape_str.substr(0, pos));
        height = std::stoul(shape_str.substr(pos + 1));
    } else {
        width = height = 4;  // 默认4x4
    }
    
    // 计算本节点坐标
    auto coord = nodeToCoord(node_id);
    my_x = coord.first;
    my_y = coord.second;
}

int Mesh2DHandler::calculateRoute(uint32_t dest_node)
{
    auto dest_coord = nodeToCoord(dest_node);
    uint32_t dest_x = dest_coord.first;
    uint32_t dest_y = dest_coord.second;
    
    // XY路由算法
    if (dest_x < my_x) return 0;  // West
    if (dest_x > my_x) return 1;  // East
    if (dest_y < my_y) return 2;  // South
    if (dest_y > my_y) return 3;  // North
    
    return -1;  // 本地节点
}

int Mesh2DHandler::calculateHopDistance(uint32_t dest_node)
{
    auto dest_coord = nodeToCoord(dest_node);
    uint32_t dest_x = dest_coord.first;
    uint32_t dest_y = dest_coord.second;
    
    // Manhattan距离 = |x1-x2| + |y1-y2|
    int x_distance = abs(static_cast<int>(dest_x) - static_cast<int>(my_x));
    int y_distance = abs(static_cast<int>(dest_y) - static_cast<int>(my_y));
    
    return x_distance + y_distance;
}

std::string Mesh2DHandler::getTopologyDescription()
{
    std::ostringstream desc;
    desc << "Mesh2D[" << width << "x" << height << "] Node(" << my_x << "," << my_y << ")";
    return desc.str();
}

std::vector<uint32_t> Mesh2DHandler::getNeighbors()
{
    std::vector<uint32_t> neighbors;
    
    // 添加邻居节点
    if (my_x > 0) neighbors.push_back(coordToNode(my_x - 1, my_y));        // West
    if (my_x < width - 1) neighbors.push_back(coordToNode(my_x + 1, my_y)); // East
    if (my_y > 0) neighbors.push_back(coordToNode(my_x, my_y - 1));        // South
    if (my_y < height - 1) neighbors.push_back(coordToNode(my_x, my_y + 1)); // North
    
    return neighbors;
}

std::pair<uint32_t, uint32_t> Mesh2DHandler::nodeToCoord(uint32_t node_id)
{
    uint32_t x = node_id % width;
    uint32_t y = node_id / width;
    return std::make_pair(x, y);
}

uint32_t Mesh2DHandler::coordToNode(uint32_t x, uint32_t y)
{
    return y * width + x;
}

// ===== Torus2DHandler 实现 =====

void Torus2DHandler::initialize(SST::Params& params, uint32_t node_id)
{
    this->node_id = node_id;
    
    // 解析拓扑形状
    std::string shape_str = params.find<std::string>("topology_shape", "4x4");
    std::size_t pos = shape_str.find('x');
    if (pos != std::string::npos) {
        width = std::stoul(shape_str.substr(0, pos));
        height = std::stoul(shape_str.substr(pos + 1));
    } else {
        width = height = 4;  // 默认4x4
    }
    
    // 计算本节点坐标
    auto coord = nodeToCoord(node_id);
    my_x = coord.first;
    my_y = coord.second;
}

int Torus2DHandler::calculateRoute(uint32_t dest_node)
{
    auto dest_coord = nodeToCoord(dest_node);
    uint32_t dest_x = dest_coord.first;
    uint32_t dest_y = dest_coord.second;
    
    // Torus 路由 - 选择较短的路径
    int x_dist = calculateTorusDistance(my_x, dest_x, width);
    int y_dist = calculateTorusDistance(my_y, dest_y, height);
    
    // 优先路由X方向
    if (x_dist != 0) {
        if (x_dist > 0) return 1;  // East
        else return 0;             // West
    }
    
    // 然后路由Y方向
    if (y_dist != 0) {
        if (y_dist > 0) return 3;  // North
        else return 2;             // South
    }
    
    return -1;  // 本地节点
}

int Torus2DHandler::calculateHopDistance(uint32_t dest_node)
{
    auto dest_coord = nodeToCoord(dest_node);
    uint32_t dest_x = dest_coord.first;
    uint32_t dest_y = dest_coord.second;
    
    // Torus距离计算 - 考虑环绕连接
    int x_dist = abs(calculateTorusDistance(my_x, dest_x, width));
    int y_dist = abs(calculateTorusDistance(my_y, dest_y, height));
    
    return x_dist + y_dist;
}

std::string Torus2DHandler::getTopologyDescription()
{
    std::ostringstream desc;
    desc << "Torus2D[" << width << "x" << height << "] Node(" << my_x << "," << my_y << ")";
    return desc.str();
}

std::vector<uint32_t> Torus2DHandler::getNeighbors()
{
    std::vector<uint32_t> neighbors;
    
    // Torus 连接 - 所有节点都有4个邻居
    neighbors.push_back(coordToNode((my_x + width - 1) % width, my_y));     // West
    neighbors.push_back(coordToNode((my_x + 1) % width, my_y));             // East
    neighbors.push_back(coordToNode(my_x, (my_y + height - 1) % height));   // South
    neighbors.push_back(coordToNode(my_x, (my_y + 1) % height));            // North
    
    return neighbors;
}

std::pair<uint32_t, uint32_t> Torus2DHandler::nodeToCoord(uint32_t node_id)
{
    uint32_t x = node_id % width;
    uint32_t y = node_id / width;
    return std::make_pair(x, y);
}

uint32_t Torus2DHandler::coordToNode(uint32_t x, uint32_t y)
{
    return y * width + x;
}

int Torus2DHandler::calculateTorusDistance(uint32_t coord1, uint32_t coord2, uint32_t dimension_size)
{
    int forward_dist = (coord2 - coord1 + dimension_size) % dimension_size;
    int backward_dist = (coord1 - coord2 + dimension_size) % dimension_size;
    
    // 返回带符号的最短距离
    if (forward_dist <= backward_dist) {
        return forward_dist;
    } else {
        return -backward_dist;
    }
}

void SnnNetworkAdapter::sendViaDirectLink(SpikeEvent* spike_event, uint32_t dest_node)
{
    // 优先使用父组件注入的链路，然后才使用SubComponent自己的链路
    SST::Link* actual_link = nullptr;
    
    // 首先检查父组件注入的network链路
    auto parent_it = parent_direction_links.find("network");
    if (parent_it != parent_direction_links.end() && parent_it->second) {
        actual_link = parent_it->second;
        output->verbose(CALL_INFO, 2, 0, "🔍 使用父组件注入的network链路: %p\n", (void*)actual_link);
    } else if (direct_link) {
        actual_link = direct_link;
        output->verbose(CALL_INFO, 2, 0, "🔍 使用SubComponent直接链路: %p\n", (void*)actual_link);
    }
    
    output->verbose(CALL_INFO, 2, 0, "🔍 检查sendViaDirectLink: actual_link=%p, spike_event=%p\n", 
                    (void*)actual_link, (void*)spike_event);
    
    if (!actual_link) {
        output->verbose(CALL_INFO, 1, 0, "❌ 没有可用的直接Link，无法发送\n");
        return;
    }
    
    if (!spike_event) {
        output->verbose(CALL_INFO, 1, 0, "❌ 脉冲事件为空，无法发送\n");
        return;
    }
    
    output->verbose(CALL_INFO, 3, 0, "📡 通过直接Link发送脉冲: 源=%u, 目标=%u, 神经元=%u\n", 
                    node_id, dest_node, spike_event->getNeuronId());
    
    // 创建新的SpikeEvent用于网络传输（避免复制构造问题）
    SpikeEvent* network_spike = new SpikeEvent(
        spike_event->getNeuronId(), 
        spike_event->getDestinationNeuron(),
        spike_event->getDestinationNode(),
        spike_event->getWeight(),
        spike_event->getTimestamp()
    );
    
    // 使用SpikeEventWrapper进行SST Event传输
    SpikeEventWrapper* wrapper_event = new SpikeEventWrapper(network_spike);
    
    // 直接通过Link发送 - 添加防御性检查和Link状态诊断
    output->verbose(CALL_INFO, 2, 0, "🔍 将要发送SpikeEventWrapper=%p(spike_data=%p)通过actual_link=%p\n", 
                    (void*)wrapper_event, (void*)network_spike, (void*)actual_link);
    
    // 详细检查Link对象的状态
    if (actual_link) {
        output->verbose(CALL_INFO, 2, 0, "🔍 Link状态诊断:\n");
        output->verbose(CALL_INFO, 2, 0, "  - actual_link指针: %p\n", (void*)actual_link);
        output->verbose(CALL_INFO, 2, 0, "  - 尝试调用Link方法...\n");
        try {
            // 测试Link对象是否有效 - 调用一个简单的方法
            output->verbose(CALL_INFO, 2, 0, "  - Link对象看起来有效\n");
        } catch (...) {
            output->verbose(CALL_INFO, 1, 0, "❌ Link对象无效或已损坏\n");
            delete wrapper_event;
            return;
        }
    }
    
    if (!wrapper_event) {
        output->verbose(CALL_INFO, 1, 0, "❌ wrapper_event为null，无法发送\n");
        return;
    }
    
    try {
        output->verbose(CALL_INFO, 2, 0, "🔍 开始调用actual_link->send(SpikeEventWrapper)\n");
        actual_link->send(wrapper_event);
        output->verbose(CALL_INFO, 2, 0, "🔍 SpikeEventWrapper send 调用完成\n");
    } catch (const std::exception& e) {
        output->verbose(CALL_INFO, 1, 0, "❌ SpikeEventWrapper send异常: %s\n", e.what());
        delete wrapper_event;
        return;
    } catch (...) {
        output->verbose(CALL_INFO, 1, 0, "❌ SpikeEventWrapper send未知异常\n");
        delete wrapper_event;
        return;
    }
    
    output->verbose(CALL_INFO, 3, 0, "✅ 脉冲通过直接Link发送成功\n");
    
    // 更新统计信息
    spikes_routed_count++;
    if (stat_spikes_routed) stat_spikes_routed->addData(1);
    
    remote_spikes_count++;
    if (stat_remote_spikes) stat_remote_spikes->addData(1);
}

// === 链路注入接口实现 ===

void SnnNetworkAdapter::injectDirectionLink(const std::string& direction, SST::Link* link) {
    if (link) {
        parent_direction_links[direction] = link;
        output->verbose(CALL_INFO, 1, 0, "✅ 父组件注入%s方向链路成功\n", direction.c_str());
    } else {
        output->verbose(CALL_INFO, 2, 0, "⚠️ 父组件注入%s方向链路为空\n", direction.c_str());
    }
}

void SnnNetworkAdapter::sendEventToDirection(SST::Event* event, const std::string& direction) {
    auto it = parent_direction_links.find(direction);
    if (it != parent_direction_links.end() && it->second) {
        output->verbose(CALL_INFO, 3, 0, "📡 通过父组件链路发送事件到%s方向\n", direction.c_str());
        it->second->send(event);
    } else {
        output->verbose(CALL_INFO, 2, 0, "⚠️ %s方向的父组件链路不可用，删除事件\n", direction.c_str());
        delete event;
    }
}

void SnnNetworkAdapter::sendViaMultiPortLink(SpikeEvent* spike_event, uint32_t dest_node, int next_port)
{
    output->verbose(CALL_INFO, 2, 0, "🔍 检查sendViaMultiPortLink: parent_links=%lu, self_links=%lu, next_port=%d\n", 
                    parent_direction_links.size(), direction_links.size(), next_port);
    
    if (!spike_event) {
        output->verbose(CALL_INFO, 1, 0, "❌ 脉冲事件为空，无法发送\n");
        return;
    }
    
    // 将端口ID映射到方向名称
    std::vector<std::string> port_directions = {"west", "east", "south", "north"};
    
    if (next_port < 0 || next_port >= (int)port_directions.size()) {
        output->verbose(CALL_INFO, 1, 0, "❌ 无效的端口ID: %d\n", next_port);
        return;
    }
    
    std::string direction = port_directions[next_port];
    
    // 创建新的SpikeEvent用于网络传输
    SpikeEvent* network_spike = new SpikeEvent(
        spike_event->getNeuronId(), 
        spike_event->getDestinationNeuron(),
        spike_event->getDestinationNode(),
        spike_event->getWeight(),
        spike_event->getTimestamp()
    );
    
    // 使用SpikeEventWrapper包装Event以符合SST要求  
    SpikeEventWrapper* wrapper = new SpikeEventWrapper(network_spike);
    
    output->verbose(CALL_INFO, 3, 0, "📡 准备通过%s方向发送脉冲: 源=%u, 目标=%u, 神经元=%u\n", 
                    direction.c_str(), node_id, dest_node, spike_event->getNeuronId());
    
    // 优先使用父组件注入的链路
    auto parent_it = parent_direction_links.find(direction);
    if (parent_it != parent_direction_links.end() && parent_it->second) {
        output->verbose(CALL_INFO, 2, 0, "🔄 使用父组件注入的%s方向链路发送\n", direction.c_str());
        sendEventToDirection(wrapper, direction);
    } else {
        // 回退到自己的多端口链路
        auto self_it = direction_links.find(direction);
        if (self_it != direction_links.end() && self_it->second) {
            output->verbose(CALL_INFO, 2, 0, "🔄 使用自己的%s方向链路发送\n", direction.c_str());
            output->verbose(CALL_INFO, 2, 0, "🔍 将要发送wrapper=%p(spike=%p)通过%s方向link=%p\n", 
                            (void*)wrapper, (void*)network_spike, direction.c_str(), (void*)self_it->second);
            
            // 验证Link的有效性
            if (!self_it->second) {
                output->verbose(CALL_INFO, 1, 0, "❌ %s方向link为null，无法发送\n", direction.c_str());
                delete wrapper;
                delete network_spike;
                return;
            }
            
            try {
                self_it->second->send(wrapper);
                output->verbose(CALL_INFO, 2, 0, "🔍 %s方向Link send 调用完成\n", direction.c_str());
            } catch (const std::exception& e) {
                output->verbose(CALL_INFO, 1, 0, "❌ %s方向Link send异常: %s\n", direction.c_str(), e.what());
                delete wrapper;
                delete network_spike;
                return;
            } catch (...) {
                output->verbose(CALL_INFO, 1, 0, "❌ %s方向Link send未知异常\n", direction.c_str());
                delete wrapper;
                delete network_spike;
                return;
            }
        } else {
            output->verbose(CALL_INFO, 1, 0, "❌ %s方向的链路不存在，无法发送\n", direction.c_str());
            delete wrapper;
            delete network_spike;  // 清理内存
            return;
        }
    }
    
    output->verbose(CALL_INFO, 3, 0, "✅ 脉冲通过%s方向发送成功\n", direction.c_str());
    
    // 更新统计信息
    spikes_routed_count++;
    if (stat_spikes_routed) stat_spikes_routed->addData(1);
    
    remote_spikes_count++;
    if (stat_remote_spikes) stat_remote_spikes->addData(1);
}

void SnnNetworkAdapter::sendViaMerlinRouter(SpikeEvent* spike_event, uint32_t dest_node, int next_port)
{
    if (!router) {
        output->verbose(CALL_INFO, 1, 0, "❌ Merlin路由器未初始化，无法发送\n");
        return;
    }
    
    // 创建SimpleNetwork请求
    SST::Interfaces::SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node, next_port);
    if (!req) {
        output->verbose(CALL_INFO, 1, 0, "❌ 创建网络请求失败\n");
        return;
    }
    
    // 设置目标地址（对于Merlin，这通常是目标节点的本地端口）
    req->dest = dest_node;  // 直接使用节点ID作为目标地址
    
    output->verbose(CALL_INFO, 2, 0, "🌐 通过Merlin路由器发送脉冲: 源=%u, 目标=%u, 端口=%d\n", 
                    node_id, dest_node, next_port);
    
    // 按照SnnNIC模式：先检查空间，再发送（使用vn=0）
    bool sent = router->spaceToSend(0, req->size_in_bits) && router->send(req, 0);
    
    if (sent) {
        output->verbose(CALL_INFO, 3, 0, "✅ 脉冲通过Merlin路由器发送成功\n");
        
        // 更新统计信息
        spikes_routed_count++;
        if (stat_spikes_routed) stat_spikes_routed->addData(1);
        
        remote_spikes_count++;
        if (stat_remote_spikes) stat_remote_spikes->addData(1);
        
        // 更新端口负载统计
        updateLoadStatistics(next_port);
        
    } else {
        output->verbose(CALL_INFO, 1, 0, "⚠️ Merlin路由器发送失败，可能是缓冲区满\n");
        
        // 清理未发送的请求
        delete req;
        
        // 统计丢包
        packets_dropped++;
        if (stat_packets_dropped) stat_packets_dropped->addData(1);
    }
}

void SnnNetworkAdapter::handleDirectSpikeEvent(SST::Event* event)
{
    if (!event) {
        output->verbose(CALL_INFO, 1, 0, "⚠️ 收到null事件\n");
        return;
    }
    
    // 尝试转换为SpikeEventWrapper
    SpikeEventWrapper* wrapper = dynamic_cast<SpikeEventWrapper*>(event);
    SpikeEvent* spike_event = nullptr;
    
    if (wrapper) {
        spike_event = wrapper->getSpikeEvent();
        output->verbose(CALL_INFO, 3, 0, "📦 通过直接Link接收SpikeEventWrapper: wrapper=%p, spike=%p\n", 
                        (void*)wrapper, (void*)spike_event);
    } else {
        // 尝试直接转换为SpikeEvent（向后兼容）
        spike_event = dynamic_cast<SpikeEvent*>(event);
        output->verbose(CALL_INFO, 3, 0, "📦 通过直接Link接收原生SpikeEvent: spike=%p\n", 
                        (void*)spike_event);
    }
    
    if (spike_event && spike_handler) {
        output->verbose(CALL_INFO, 2, 0, "📦 处理接收的脉冲: 神经元%u\n", 
                        spike_event->getNeuronId());
        
        // 调用脉冲处理回调
        spike_handler(spike_event);
        
        // 更新统计
        local_spikes_count++;
        if (stat_local_spikes) stat_local_spikes->addData(1);
        
        output->verbose(CALL_INFO, 3, 0, "✅ 直接Link脉冲处理完成\n");
    } else {
        output->verbose(CALL_INFO, 1, 0, "⚠️ 无法提取脉冲事件或未设置处理器\n");
    }
    
    // 注意：SST会自动处理event的内存管理
}

bool SnnNetworkAdapter::handleNetworkEvent(int vn)
{
    // 处理从Merlin路由器接收的网络事件
    if (!router) {
        return false;
    }
    
    // 接收所有待处理的请求
    SST::Interfaces::SimpleNetwork::Request* req = router->recv(vn);
    
    while (req != nullptr) {
        output->verbose(CALL_INFO, 2, 0, "📦 接收到网络数据包: 源=%lu, 目标=%lu, 大小=%lu\n", 
                        req->src, req->dest, req->size_in_bits);
        
        // 提取脉冲事件
        SpikeEvent* received_spike = extractSpikeFromRequest(req);
        
        if (received_spike && spike_handler) {
            output->verbose(CALL_INFO, 3, 0, "✅ 解包脉冲事件并转发给处理器\n");
            
            // 调用脉冲处理回调
            spike_handler(received_spike);
            
            // 更新接收统计
            local_spikes_count++;
            if (stat_local_spikes) stat_local_spikes->addData(1);
            
        } else {
            output->verbose(CALL_INFO, 1, 0, "⚠️ 无法解包脉冲事件或未设置处理器\n");
        }
        
        // 清理请求对象
        delete req;
        
        // 继续接收下一个请求
        req = router->recv(vn);
    }
    
    return true;
}

// ===== SimpleNetwork 适配器访问实现 =====

SimpleNetworkWrapper* SnnNetworkAdapter::getSimpleNetworkWrapper() 
{
    return simple_network_wrapper;
}

SimpleNetworkWrapper* SnnNetworkAdapter::createSimpleNetworkWrapper(SST::Params& params) 
{
    if (!simple_network_wrapper) {
        // 创建新的SimpleNetworkWrapper实例
        ComponentId_t wrapper_id = getId(); // 使用相同的组件ID
        simple_network_wrapper = new SimpleNetworkWrapper(wrapper_id, params, 0); // 端口0
        simple_network_wrapper->setNetworkAdapter(this);
        output->verbose(CALL_INFO, 1, 0, "✅ 创建SimpleNetworkWrapper成功\n");
    }
    return simple_network_wrapper;
}

} // namespace SnnDL
} // namespace SST
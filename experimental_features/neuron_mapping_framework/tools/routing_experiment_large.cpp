#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "routing/RoutingTableGenerator.h"
#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include <chrono>

using namespace NeuronMapping;

static void write_text(const std::string& path, const std::string& s) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ofs << s;
}

int main(int argc, char** argv) {
    // 参数：N outdeg rows cols per_pe seed
    uint32_t N = 65536;          // 神经元总数
    uint32_t outdeg = 64;        // 每个神经元期望出度
    uint32_t rows = 8, cols = 8; // 8x8 PE
    uint32_t per_pe = 1024;      // 每PE神经元容量（N / (rows*cols)）
    uint32_t seed = 42;
    if (argc >= 2) N = static_cast<uint32_t>(std::stoul(argv[1]));
    if (argc >= 3) outdeg = static_cast<uint32_t>(std::stoul(argv[2]));
    if (argc >= 5) { rows = static_cast<uint32_t>(std::stoul(argv[3])); cols = static_cast<uint32_t>(std::stoul(argv[4])); }
    if (argc >= 6) per_pe = static_cast<uint32_t>(std::stoul(argv[5]));
    if (argc >= 7) seed = static_cast<uint32_t>(std::stoul(argv[6]));

    const uint32_t num_pes = rows * cols;
    if (per_pe * num_pes < N) {
        std::cerr << "容量不足: per_pe*PEs < N\n"; return 1;
    }

    std::cout << "== 大规模路由实验 ==\n";
    std::cout << "neurons=" << N << ", outdeg=" << outdeg << ", topo=" << rows << "x" << cols
              << ", per_pe=" << per_pe << ", seed=" << seed << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1) 生成神经网络
    NeuralNetwork net;
    for (uint32_t i = 0; i < N; ++i) net.addNeuron(NeuronProperties(i));

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> idist(0, N - 1);
    std::normal_distribution<float> wdist(0.1f, 0.05f); // 权重均值0.1，std=0.05
    std::uniform_real_distribution<float> ddelay(0.5f, 2.0f); // 0.5~2.0ms

    // 稀疏随机连边：每个神经元 outdeg 个随机目标（避免自环）
    for (uint32_t src = 0; src < N; ++src) {
        for (uint32_t k = 0; k < outdeg; ++k) {
            uint32_t tgt = idist(rng);
            if (tgt == src) { if (tgt + 1 < N) tgt++; else if (tgt > 0) tgt--; }
            float w = std::max(0.01f, wdist(rng));
            net.addConnection(Connection(src, tgt, w));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "网络生成完成: connections≈" << (static_cast<uint64_t>(N) * outdeg) << "\n";

    // 2) 生成拓扑 8x8 Mesh
    HardwareTopology topo;
    ProcessingElement pe_cfg; pe_cfg.max_neurons = per_pe; pe_cfg.memory_capacity = 64ull * 1024ull * 1024ull; // 64MB占位
    topo.createMesh2D(rows, cols, pe_cfg);

    // 3) 图分割映射
    neuron_mapping::GraphPartitioningStrategy part;
    MappingConfig mcfg; // 默认
    auto t2 = std::chrono::high_resolution_clock::now();
    auto mapping = part.mapNetwork(net, topo, mcfg);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (!mapping) { std::cerr << "映射失败\n"; return 2; }

    // 3.1) 按每PE 8个核心、每核心最多128个神经元，分配 core_id
    {
        std::unordered_map<PEId, std::vector<NeuronId>> pe_to_neurons;
        for (const auto& asg : mapping->getAllAssignments()) pe_to_neurons[asg.pe_id].push_back(asg.neuron_id);
        for (auto& kv : pe_to_neurons) {
            auto& vec = kv.second;
            std::sort(vec.begin(), vec.end()); // 稳定排序
            for (size_t idx = 0; idx < vec.size(); ++idx) {
                uint32_t core_id = static_cast<uint32_t>(idx / 128u);
                if (core_id >= 8u) core_id = 7u; // 尽量避免越界（理论上不会超过1024/128=8）
                mapping->reassignNeuron(vec[idx], kv.first, core_id);
            }
        }
    }

    // 4) 路由生成（启用多播+连续LSB前缀聚合）
    RoutingGenerationConfig rcfg;
    rcfg.enable_multicast = true;
    rcfg.enable_compression = true;
    rcfg.compression = CompressionMethod::PREFIX_AGGREGATION;
    RoutingTableGenerator gen(rcfg);
    auto t4 = std::chrono::high_resolution_clock::now();
    auto tables = gen.generateRoutingTables(net, topo, *mapping, rcfg);
    auto t5 = std::chrono::high_resolution_clock::now();

    // 5) 仅导出汇总报告和每PE表项行数（避免巨大的JSON路由表）
    std::filesystem::create_directories("export_large");
    std::string csv = gen.exportVisualizationData(tables, topo, "csv");
    write_text("export_large/routing_stats.csv", csv);
    std::string report = gen.generateRoutingReport(tables, "json");
    write_text("export_large/report.json", report);

    // 附加运行时间信息
    auto dur_gen_net = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto dur_map = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    auto dur_route = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
    std::ostringstream ext;
    ext << "{\"gen_net_ms\":" << dur_gen_net
        << ",\"map_ms\":" << dur_map
        << ",\"route_ms\":" << dur_route << "}";
    write_text("export_large/timings.json", ext.str());

    std::cout << "导出完成：export_large/report.json, routing_stats.csv, timings.json\n";
    return 0;
}

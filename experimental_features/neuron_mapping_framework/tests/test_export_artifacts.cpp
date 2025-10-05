#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "routing/RoutingTableGenerator.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace NeuronMapping;

static bool file_nonempty(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return false;
    return std::filesystem::file_size(path, ec) > 0;
}

int main() {
    std::cout << "== exportArtifacts API 测试 ==\n";

    // 1) 构造小网络：16神经元，两层全连接
    NeuralNetwork net;
    const uint32_t N = 16, L = 2, per = N / L;
    for (uint32_t i = 0; i < N; ++i) net.addNeuron(NeuronProperties(i));
    for (uint32_t i = 0; i < per; ++i)
        for (uint32_t j = 0; j < per; ++j)
            net.addConnection(Connection(i, per + j, 0.2f));

    // 2) 构造2x2 Mesh拓扑（4 PEs）
    HardwareTopology topo;
    ProcessingElement pe_cfg; pe_cfg.max_neurons = per; pe_cfg.memory_capacity = 1024 * 1024;
    if (!topo.createMesh2D(2, 2, pe_cfg)) {
        std::cerr << "拓扑创建失败\n"; return 2;
    }

    // 3) 映射（图分割）
    neuron_mapping::GraphPartitioningStrategy part;
    MappingConfig mcfg; // 默认
    auto mapping = part.mapNetwork(net, topo, mcfg);
    if (!mapping) { std::cerr << "映射失败\n"; return 3; }

    // 4) 生成并导出
    RoutingGenerationConfig rcfg; rcfg.enable_multicast = true; rcfg.enable_compression = true;
    RoutingTableGenerator gen(rcfg);
    const std::string outdir = "export_test";
    std::filesystem::remove_all(outdir);
    bool ok = gen.exportArtifacts(net, topo, *mapping, outdir, rcfg);
    if (!ok) { std::cerr << "导出失败\n"; return 4; }

    // 5) 验证产物存在且非空
    bool has_mapping = file_nonempty(outdir + "/mapping.json");
    bool has_tables  = file_nonempty(outdir + "/routing_tables.json");
    bool has_stats   = file_nonempty(outdir + "/routing_stats.csv");
    bool has_report  = file_nonempty(outdir + "/report.json");

    if (!(has_mapping && has_tables && has_stats && has_report)) {
        std::cerr << "产物缺失: mapping=" << has_mapping << ", tables=" << has_tables
                  << ", stats=" << has_stats << ", report=" << has_report << "\n";
        return 5;
    }

    // 6) 简单解析 report.json，检查压缩比 < 1 且 total_entries == sum(csv)
    std::ifstream rifs(outdir + "/report.json");
    std::stringstream rbuf; rbuf << rifs.rdbuf();
    std::string report_json = rbuf.str();

    auto find_number = [&](const std::string& key, float& out){
        auto pos = report_json.find(key);
        if (pos == std::string::npos) return false;
        pos = report_json.find(':', pos);
        if (pos == std::string::npos) return false;
        size_t end = pos + 1;
        while (end < report_json.size() && (report_json[end] == ' ')) ++end;
        size_t start = end;
        while (end < report_json.size() && (isdigit(report_json[end]) || report_json[end]=='.')) ++end;
        try { out = std::stof(report_json.substr(start, end - start)); } catch(...) { return false; }
        return true;
    };
    float compression_ratio = 1.0f, total_entries_rep = 0.0f;
    bool ok_cr = find_number("\"compression_ratio\"", compression_ratio);
    bool ok_te = find_number("\"total_entries\"", total_entries_rep);

    // 解析 csv 汇总 entries
    std::ifstream cifs(outdir + "/routing_stats.csv");
    std::string line; std::getline(cifs, line); // header
    int sum_csv = 0; while (std::getline(cifs, line)) { if (line.empty()) continue; auto comma = line.find(','); if (comma!=std::string::npos) { int n = std::stoi(line.substr(comma+1)); sum_csv += n; } }

    if (!(ok_cr && ok_te && compression_ratio <= 1.0f && static_cast<int>(total_entries_rep) == sum_csv)) {
        std::cerr << "报告校验失败: compression_ratio=" << compression_ratio << ", total_entries(rep)=" << total_entries_rep << ", sum_csv=" << sum_csv << "\n";
        return 6;
    }

    std::cout << "✓ 导出产物存在且非空，压缩比=" << compression_ratio << ", total_entries=" << sum_csv << "\n";
    std::cout << "输出目录: " << outdir << "\n";
    return 0;
}

// -*- c++ -*-
//
// WeightLoader.h: 在init阶段通过StandardMem将权重写入内存的组件
//

#ifndef _SNNDL_WEIGHT_LOADER_H
#define _SNNDL_WEIGHT_LOADER_H

#include <sst/core/component.h>
#include <sst/core/output.h>
#include <sst/core/interfaces/stdMem.h>

namespace SST {
namespace SnnDL {

class WeightLoader : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        WeightLoader,
        "SnnDL",
        "WeightLoader",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Init阶段从文件加载权重并写入内存",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "日志详细级别", "0"},
        {"weight_file", "兼容旧参数：单文件路径(若提供将优先生效)", ""},
        {"base_addr_start", "core0权重矩阵的起始地址", "0"},
        {"per_core_stride", "相邻核心权重矩阵在内存中的地址跨度(字节)", "0"},
        {"num_cores", "核心数", "1"},
        {"neurons_per_core", "每核神经元数(形成 NxN 权重矩阵)", "64"},
        {"fill_value", "当无文件可用时使用的填充值(float)", "0.5"},
        {"weight_format", "权重文件格式: bin/csv", "bin"},
        {"per_core_files", "是否按核心分文件(1=是,0=否)", "0"},
        {"file_template", "按核心分文件时的模板, 例如 weights_core{core}.bin", ""},
        {"single_file", "单文件路径(覆盖weight_file)", ""},
        {"row_major", "文件是否按行优先(1=是,0=否=列优先)", "1"},
        {"chunk_size_bytes", "每次写入的字节块大小(建议与cacheline一致)", "64"},
        {"validate_length", "是否校验文件长度与期望匹配", "1"}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory", "StandardMem内存接口", "SST::Interfaces::StandardMem"}
    )

    WeightLoader(SST::ComponentId_t id, SST::Params& params);
    ~WeightLoader() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;
    void handleMemoryResponse(SST::Interfaces::StandardMem::Request* req);

private:
    void loadFileOnce();
    void loadFileOnceRuntime();
    void issueWritesFill(float value);
    void issueWritesForCoreFloats(int core, const std::vector<float>& wbuf);
    void issueWritesForCoreFloatsRuntime(int core, const std::vector<float>& wbuf);
    bool readFileAllFloats(const std::string& path, const std::string& fmt, std::vector<float>& out);
    bool loadSingleFileAllCores(const std::string& path, const std::string& fmt);
    bool loadPerCoreFiles(const std::string& tmpl, const std::string& fmt);
    bool loadPerCoreFilesRuntime(const std::string& tmpl, const std::string& fmt);

    SST::Output* output_;
    SST::Interfaces::StandardMem* memory_;

    int verbose_;
    std::string weight_file_;
    uint64_t base_addr_start_;
    uint64_t per_core_stride_;
    int num_cores_;
    uint32_t neurons_per_core_;
    float fill_value_;
    std::string weight_format_;
    bool per_core_files_;
    std::string file_template_;
    std::string single_file_;
    bool row_major_;
    uint32_t chunk_size_bytes_;
    bool validate_length_;
    int file_core_offset_ = 0; // 读取单文件时偏移的核心数

    // Timed seed writes to ensure visibility in timed simulation
    bool timed_seed_enable_ = true;
    uint32_t timed_seed_count_ = 1; // per core
    bool seed_done_ = false;
    bool clock_registered_ = false;
    SST::Cycle_t current_cycle_ = 0;

    bool onClockTick(SST::Cycle_t cycle);

    bool loaded_;
    bool runtime_load_needed_ = false;
    uint32_t pending_writes_ = 0;
    bool all_writes_completed_ = false;
};

}
}

#endif


// -*- c++ -*-
//
// WeightLoader.h: 在init阶段通过StandardMem将权重写入内存的组件
//

#ifndef _SNNDL_WEIGHT_LOADER_H
#define _SNNDL_WEIGHT_LOADER_H

#include <sst/core/component.h>
#include <sst/core/output.h>
#include <sst/core/interfaces/stdMem.h>
#include <unordered_map>

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
        {"rows_per_core", "每核权重矩阵的行数(默认等于neurons_per_core)", "0"},
        {"cols_per_core", "每核权重矩阵的列数(默认等于neurons_per_core)", "0"},
        {"fill_value", "当无文件可用时使用的填充值(float)", "0.5"},
        {"weight_format", "权重文件格式: bin/csv/raw", "bin"},
        {"per_core_files", "是否按核心分文件(1=是,0=否)", "0"},
        {"file_template", "按核心分文件时的模板, 例如 weights_core{core}.bin", ""},
        {"single_file", "单文件路径(覆盖weight_file)", ""},
        {"row_major", "文件是否按行优先(1=是,0=否=列优先)", "1"},
        {"chunk_size_bytes", "每次写入的字节块大小(建议与cacheline一致)", "64"},
        {"validate_length", "是否校验文件长度与期望匹配", "1"},
        // 运行时种子写入：默认关闭；若开启但连接路径经过缓存，建议保持关闭以避免coherence拒绝Write
        {"timed_seed_enable", "是否在运行时第一个周期进行计时写入(1/0)", "1"},
        {"timed_seed_count", "每个核心用于种子写入的块数（行粒度），默认1", "1"},
        {"timed_seed_allow_cache", "允许通过缓存层进行计时写入(0=默认不允许，避免Write被coherence拒绝)", "0"}
        ,
        // BCSR 支持（可选）：当 weight_format=bcsr 或 bcsr_enable=1 时生效
        {"bcsr_enable", "启用BCSR权重写入(1/0)", "0"},
        {"bcsr_block_rows", "BCSR块行数br（需整除rows_per_core）", "16"},
        {"bcsr_block_cols", "BCSR块列数bc（需整除cols_per_core）", "16"},
        {"bcsr_val_bytes", "数值字节数（默认4=FP32）", "4"},
        {"bcsr_idx_bytes", "索引字节数（默认2=uint16）", "2"},
        {"bcsr_pattern", "BCSR生成模式：diag|full（默认diag，仅对fill模式）", "diag"}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory", "StandardMem内存接口", "SST::Interfaces::StandardMem"}
    )

    // 统计信息文档（Batch‑B: WeightLoader 写入画像）
    SST_ELI_DOCUMENT_STATISTICS(
        {"weight_bytes_written_total", "总写入字节（所有块累计）", "bytes", 1},
        {"weight_write_chunks_total", "写入块总数（所有chunk累计）", "chunks", 1},
        {"weight_write_chunk_bytes", "单次写入字节直方图", "bytes", 1},
        {"weight_write_latency_cycles", "写入端到端时延直方图（仅timed模式）", "cycles", 1},
        {"weight_write_total_cycles", "本次写入总耗时（cycle，仅timed模式，上报一次）", "cycles", 1}
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
    // Shared row-write implementation; timed=false uses sendUntimedData with posted=true,
    // timed=true uses timed send() with posted=false and tracks pending_writes_.
    void issueWritesForCoreFloatsImpl(int core, const std::vector<float>& wbuf, bool timed);
    void issueWritesForCoreFloats(int core, const std::vector<float>& wbuf);
    void issueWritesForCoreFloatsRuntime(int core, const std::vector<float>& wbuf);
    bool readFileAllFloats(const std::string& path, const std::string& fmt, std::vector<float>& out);
    bool readFileRaw(const std::string& path, std::vector<uint8_t>& out);
    void issueWritesRaw(int core, const std::vector<uint8_t>& data, bool timed);
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
    // 新增：显式行列配置，默认退回到neurons_per_core_（保持兼容）
    uint32_t rows_per_core_;
    uint32_t cols_per_core_;
    float fill_value_;
    std::string weight_format_;
    bool raw_mode_ = false;
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
    bool timed_seed_allow_cache_ = false; // 默认不通过缓存执行写入
    bool seed_done_ = false;
    bool clock_registered_ = false;
    SST::Cycle_t current_cycle_ = 0;
    // 写入统计（Batch‑B）
    Statistic<uint64_t>* stat_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_chunks_total_ = nullptr;
    Statistic<uint64_t>* stat_chunk_bytes_hist_ = nullptr;
    Statistic<uint64_t>* stat_write_latency_cycles_hist_ = nullptr;
    Statistic<uint64_t>* stat_write_total_cycles_ = nullptr;
    // 仅timed模式：记录发起周期与总耗时
    bool write_started_timed_ = false;
    uint64_t write_first_issue_cycle_ = 0;
    std::unordered_map<SST::Interfaces::StandardMem::Request::id_t, uint64_t> write_issue_cycle_;

    bool onClockTick(SST::Cycle_t cycle);
    // 参数规范化与健壮性检查
    void normalizeParams_();
    // 统一的分块写入帮助函数（带统计）
    inline void sendWrite_(uint64_t addr, const std::vector<uint8_t>& payload, bool timed) {
        auto* w = new SST::Interfaces::StandardMem::Write(addr, payload.size(), payload, !timed);
        if (timed) {
            // 记录发起周期（用于时延统计）
            write_issue_cycle_[w->getID()] = current_cycle_;
            memory_->send(w);
            pending_writes_++;
        } else {
            memory_->sendUntimedData(w);
        }
        // 统计：块大小/块数/字节数
        if (stat_chunk_bytes_hist_) stat_chunk_bytes_hist_->addData(static_cast<uint64_t>(payload.size()));
        if (stat_chunks_total_) stat_chunks_total_->addData(1);
        if (stat_bytes_total_) stat_bytes_total_->addData(static_cast<uint64_t>(payload.size()));
    }

    bool loaded_;
    bool runtime_load_needed_ = false;
    uint32_t pending_writes_ = 0;
    bool all_writes_completed_ = false;

    // BCSR 配置
    bool bcsr_enable_ = false;
    uint32_t bcsr_br_ = 16;
    uint32_t bcsr_bc_ = 16;
    uint32_t bcsr_val_bytes_ = 4;
    uint32_t bcsr_idx_bytes_ = 2;
    std::string bcsr_pattern_ = "diag";
    void issueWritesBCSRFill(float value);
};

}
}

#endif

// -*- c++ -*-
//
// WorkloadConfig: 统一的 workload 选择入口（集中读取环境变量并缓存）
// - 仅用于“保持脚本不改”的 Phase6 兼容路径
// - 读一次 env，后续 O(1) 查询，避免在热路径重复 getenv/strcmp
//

#pragma once

#include <cstdlib>
#include <cstring>

namespace SST { namespace SnnDL {

inline const char* workloadImplFromEnvCached() {
    static const char* cached = []() -> const char* {
        const char* v = std::getenv("SNNDL_WORKLOAD_IMPL");
        if (!v || !*v) return nullptr;
        return v;
    }();
    return cached;
}

inline bool isStreamWorkloadEnv() {
    const char* v = workloadImplFromEnvCached();
    return v && std::strcmp(v, "stream") == 0;
}

}} // namespace SST::SnnDL


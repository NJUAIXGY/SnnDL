// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent_routing.cc: 路由构建与共享缓存逻辑拆分
//

#include <sst/core/sst_config.h>

#include "SnnPESubComponent.h"
#include "api/ISnnSpikeCommWorkload.h"
#include "synapse/weights/SnnBcsrWeightManager.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <inttypes.h>
#include <iterator>
#include <limits>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;


std::string SnnPESubComponent::resolveWeightTemplate(uint32_t pe, int core) const {
    if (weights_template_.empty()) return "";
    std::string path = weights_template_;
    auto replaceIndexed = [&](const std::string& marker, uint32_t value, int width) {
        size_t pos = 0;
        while ((pos = path.find(marker, pos)) != std::string::npos) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%0*u", width, value);
            path.replace(pos, marker.size(), buf);
            pos += width;
        }
    };
    auto replaceSimple = [&](const std::string& marker, uint32_t value) {
        size_t pos = 0;
        std::string text = std::to_string(value);
        while ((pos = path.find(marker, pos)) != std::string::npos) {
            path.replace(pos, marker.size(), text);
            pos += text.size();
        }
    };
    replaceIndexed("{pe:02d}", pe, 2);
    replaceSimple("{pe}", pe);
    replaceIndexed("{core:02d}", static_cast<uint32_t>(core), 2);
    replaceSimple("{core}", static_cast<uint32_t>(core));
    return path;
}

void SnnPESubComponent::applyGatingDecision(uint32_t src_global, const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle, uint64_t ttl_cycles)
{
    // Phase4-Task6.3：门控决策的应用下沉到 workload=snn（synapse/route）。
    if (snn_comm_workload_) {
        snn_comm_workload_->applyGatingDecision(src_global, dest_pes, current_cycle, ttl_cycles);
    }
}

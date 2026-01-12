// -*- c++ -*-
//
// GlobalGasStepController implementation
//

#include "gas/GlobalGasStepController.h"

#include <string>

#include "GasStepBarrierEvent.h"

namespace SST { namespace SnnDL {

GlobalGasStepController::GlobalGasStepController(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
{
    verbose_ = params.find<int>("verbose", 0);
    out_ = new SST::Output("GlobalGasStepController[@p:@l]: ", verbose_, 0, SST::Output::STDOUT);

    start_seq_ = params.find<uint32_t>("start_seq", 1);
    max_steps_ = params.find<uint32_t>("max_steps", 0);
    require_all_ready_ = params.find<int>("require_all_ready", 1) != 0;
    strict_seq_check_ = params.find<int>("strict_seq_check", 1) != 0;

    // 仅在 step-limited 模式下注册为 primary，避免影响默认的 time-based stop-at 回归口径。
    primary_keepalive_ = (max_steps_ > 0);
    if (primary_keepalive_) {
        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();
    }

    // 端口向量：pe_link0, pe_link1, ...
    const std::string prefix = "pe_link";
    int idx = 0;
    std::string port = prefix + std::to_string(idx);
    while (isPortConnected(port)) {
        SST::Link* link =
            configureLink(port,
                          new SST::Event::Handler2<GlobalGasStepController, &GlobalGasStepController::handleBarrierEvent_, int>(this, idx));
        if (!link) {
            out_->fatal(CALL_INFO, -1, "GlobalGasStepController fatal: unable to configure link %s\n", port.c_str());
        }
        pe_links_.push_back(link);
        ++idx;
        port = prefix + std::to_string(idx);
    }

    if (pe_links_.empty()) {
        out_->fatal(CALL_INFO, -1, "GlobalGasStepController fatal: no PE links connected (expected pe_link0..)\n");
    }
    pe_ready_.assign(pe_links_.size(), 0);
    pe_done_.assign(pe_links_.size(), 0);
}

GlobalGasStepController::~GlobalGasStepController() {
    // SST 管理 Component 生命周期；Output 由 SST/进程退出统一回收（避免析构次序竞态）
    out_ = nullptr;
}

void GlobalGasStepController::init(unsigned int /*phase*/) {
    // no-op
}

void GlobalGasStepController::setup() {
    // no-op
}

void GlobalGasStepController::finish() {
    if (!out_) return;
    size_t ready = 0;
    size_t done = 0;
    for (auto v : pe_ready_) if (v) ++ready;
    for (auto v : pe_done_) if (v) ++done;
    out_->verbose(
        CALL_INFO, 1, 0,
        "[step-sync] finish: started=%d current_seq=%u max_steps=%u completed=%u ready=%zu/%zu done=%zu/%zu\n",
        started_ ? 1 : 0,
        current_seq_,
        max_steps_,
        steps_completed_,
        ready,
        pe_ready_.size(),
        done,
        pe_done_.size());
}

bool GlobalGasStepController::allReady_() const {
    for (auto v : pe_ready_) if (!v) return false;
    return true;
}

bool GlobalGasStepController::allDone_() const {
    for (auto v : pe_done_) if (!v) return false;
    return true;
}

void GlobalGasStepController::broadcastStart_(uint32_t seq) {
    for (size_t i = 0; i < pe_links_.size(); ++i) {
        auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::StartStep, seq, /*src_node*/0);
        pe_links_[i]->send(ev);
    }
    if (out_) {
        out_->verbose(CALL_INFO, 1, 0, "[step-sync] START_STEP seq=%u broadcast to %zu PEs\n", seq, pe_links_.size());
    }
}

void GlobalGasStepController::handleBarrierEvent_(SST::Event* ev, int pe_index) {
    auto* msg = dynamic_cast<GasStepBarrierEvent*>(ev);
    if (!msg) {
        delete ev;
        return;
    }
    if (pe_index < 0 || static_cast<size_t>(pe_index) >= pe_links_.size()) {
        delete msg;
        return;
    }

    const GasStepBarrierOp op = msg->operation();
    const uint32_t seq = msg->seq;
    const uint32_t src_node = msg->src_node;

    if (op == GasStepBarrierOp::PeReady) {
        pe_ready_[pe_index] = 1;
        if (out_) {
            out_->verbose(CALL_INFO, 2, 0, "[step-sync] PE_READY pe=%d src_node=%u\n", pe_index, src_node);
        }
        if (!started_ && (!require_all_ready_ || allReady_())) {
            started_ = true;
            current_seq_ = start_seq_;
            pe_done_.assign(pe_links_.size(), 0);
            broadcastStart_(current_seq_);
        }
        delete msg;
        return;
    }

    if (op == GasStepBarrierOp::PeDone) {
        if (!started_) {
            if (strict_seq_check_) {
                if (out_) out_->fatal(CALL_INFO, -1, "GlobalGasStepController fatal: PE_DONE before started (pe=%d)\n", pe_index);
            }
            delete msg;
            return;
        }
        if (seq != current_seq_) {
            if (strict_seq_check_) {
                if (out_) out_->fatal(
                    CALL_INFO, -1,
                    "GlobalGasStepController fatal: unexpected PE_DONE seq=%u current_seq=%u (pe=%d src_node=%u)\n",
                    seq, current_seq_, pe_index, src_node);
            } else {
                if (out_) out_->verbose(
                    CALL_INFO, 1, 0,
                    "[step-sync] ignore PE_DONE seq=%u current_seq=%u (pe=%d)\n",
                    seq, current_seq_, pe_index);
            }
            delete msg;
            return;
        }
        pe_done_[pe_index] = 1;
        if (out_) {
            out_->verbose(CALL_INFO, 3, 0, "[step-sync] PE_DONE seq=%u pe=%d src_node=%u\n", seq, pe_index, src_node);
        }
        if (allDone_()) {
            steps_completed_ += 1;
            if (max_steps_ > 0 && steps_completed_ >= max_steps_) {
                if (out_) {
                    out_->verbose(CALL_INFO, 1, 0,
                                  "[step-sync] reached max_steps=%u (completed=%u), stopping\n",
                                  max_steps_, steps_completed_);
                }
                if (primary_keepalive_) {
                    primaryComponentOKToEndSim();
                }
                delete msg;
                return;
            }

            ++current_seq_;
            pe_done_.assign(pe_links_.size(), 0);
            broadcastStart_(current_seq_);
        }
        delete msg;
        return;
    }

    // StartStep 不应从 PE 侧回送
    if (op == GasStepBarrierOp::StartStep) {
        if (strict_seq_check_ && out_) {
            out_->fatal(CALL_INFO, -1, "GlobalGasStepController fatal: unexpected START_STEP from PE (pe=%d)\n", pe_index);
        }
        delete msg;
        return;
    }

    delete msg;
}

}} // namespace SST::SnnDL

// -*- c++ -*-
//
// GlobalGasStepController implementation
//

#include "gas/GlobalGasStepController.h"

#include <string>

#include "GasStepBarrierEvent.h"
#include "gas/GlobalGasStepControllerConfig.h"

namespace SST { namespace SnnDL {

GlobalGasStepController::GlobalGasStepController(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id)
{
    const GlobalGasStepControllerConfig cfg = parseGlobalGasStepControllerConfig(params);
    verbose_ = cfg.verbose;
    out_ = new SST::Output("GlobalGasStepController[@p:@l]: ", verbose_, 0, SST::Output::STDOUT);

    clock_freq_ = params.find<std::string>("clock", "1GHz");
    start_seq_ = cfg.start_seq;
    max_steps_ = cfg.max_steps;
    require_all_ready_ = cfg.require_all_ready;
    strict_seq_check_ = cfg.strict_seq_check;
    if (start_seq_ == 0) {
        if (strict_seq_check_) {
            if (out_) {
                out_->fatal(CALL_INFO, -1, "GlobalGasStepController fatal: start_seq=0 is invalid (expected >=1)\n");
            }
        } else {
            if (out_) {
                out_->verbose(CALL_INFO, 1, 0, "[step-warn] start_seq=0 invalid, clamp to 1\n");
            }
            ++warn_count_;
            start_seq_ = 1;
        }
    }

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

    const int first_gap = idx;
    if (first_gap > 0) {
        const int kGapProbe = 8;
        int found = -1;
        for (int probe = first_gap + 1; probe <= first_gap + kGapProbe; ++probe) {
            const std::string probe_port = prefix + std::to_string(probe);
            if (isPortConnected(probe_port)) {
                found = probe;
                break;
            }
        }
        if (found >= 0) {
            if (strict_seq_check_) {
                out_->fatal(CALL_INFO, -1,
                            "GlobalGasStepController fatal: port gap detected (pe_link%d missing, but pe_link%d connected)\n",
                            first_gap, found);
            } else {
                if (out_) {
                    out_->verbose(CALL_INFO, 1, 0,
                                  "[step-warn] port gap: pe_link%d missing while pe_link%d connected; higher ports ignored\n",
                                  first_gap, found);
                }
                ++warn_count_;
            }
        }
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
    // Threading robustness:
    // - When running with multiple SST threads, this controller may be placed on a different thread than PEs.
    // - If the controller has no clock/self events, its thread can remain at time 0 and miss cross-thread barrier events.
    // - Register a lightweight clock to ensure the owning thread participates in time advancement.
    if (!clock_registered_) {
        registerClock(clock_freq_, new SST::Clock::Handler2<GlobalGasStepController, &GlobalGasStepController::clockTick_>(this));
        clock_registered_ = true;
    }
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
    if (warn_count_ > 0) {
        out_->verbose(
            CALL_INFO, 0, 0,
            "[step-warn] finish: warn_count=%u started=%d current_seq=%u max_steps=%u completed=%u ready=%zu/%zu done=%zu/%zu\n",
            warn_count_,
            started_ ? 1 : 0,
            current_seq_,
            max_steps_,
            steps_completed_,
            ready,
            pe_ready_.size(),
            done,
            pe_done_.size());
        return;
    }
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
            } else {
                if (out_) {
                    out_->verbose(CALL_INFO, 1, 0,
                                  "[step-warn] PE_DONE before started ignored (pe=%d src_node=%u seq=%u)\n",
                                  pe_index, src_node, seq);
                }
                ++warn_count_;
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
                    "[step-warn] unexpected PE_DONE seq=%u current_seq=%u (pe=%d src_node=%u) ignored\n",
                    seq, current_seq_, pe_index, src_node);
                ++warn_count_;
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
        } else {
            if (out_) {
                out_->verbose(CALL_INFO, 1, 0,
                              "[step-warn] unexpected START_STEP from PE ignored (pe=%d src_node=%u seq=%u)\n",
                              pe_index, src_node, seq);
            }
            ++warn_count_;
        }
        delete msg;
        return;
    }

    delete msg;
}

bool GlobalGasStepController::clockTick_(SST::Cycle_t /*cycle*/) {
    // keep ticking
    return false;
}

}} // namespace SST::SnnDL

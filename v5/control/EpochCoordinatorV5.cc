#include <sst/core/sst_config.h>

#include "EpochCoordinatorV5.h"

#include <algorithm>
#include <fstream>

namespace SST { namespace SnnDL { namespace v5 {

EpochCoordinatorV5::EpochCoordinatorV5(SST::ComponentId_t id, SST::Params& params)
    : Component(id), out_("SnnDL.EpochCoordinatorV5", 0, 0, Output::STDOUT),
      pes_(std::max(1u, params.find<std::uint32_t>("mesh_pes", 1))),
      cores_per_pe_(std::max(1u, params.find<std::uint32_t>("cores_per_pe", 1))),
      start_timestep_(params.find<std::uint64_t>("start_timestep", 0)),
      timesteps_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("timesteps", 1))),
      timeout_cycles_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("timeout_cycles", 1000000))),
      epoch_(start_timestep_), output_json_(params.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    command_ = configureLink("command");
    status_ = configureLink("status", new Event::Handler2<EpochCoordinatorV5, &EpochCoordinatorV5::handleStatus_>(this));
    if (!command_ || !status_) out_.fatal(CALL_INFO, -1, "EpochCoordinatorV5 requires command and status links\n");
    seen_.resize(static_cast<std::size_t>(pes_) * cores_per_pe_, false);
    registerClock(params.find<std::string>("clock", "1GHz"), new Clock::Handler2<EpochCoordinatorV5, &EpochCoordinatorV5::tick_>(this));
    for (const char* name : {"sync.commands", "sync.reports", "sync.epochs_completed",
                             "sync.barrier_wait_ns", "sync.timeouts"}) {
        registerStatistic<std::uint64_t>(name);
    }
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

std::size_t EpochCoordinatorV5::participant_(const NocControlV5Event& packet) const {
    if (packet.source_pe >= pes_ || packet.source_core >= cores_per_pe_) {
        out_.fatal(CALL_INFO, -1, "epoch report has invalid PE/Core source\n");
    }
    return static_cast<std::size_t>(packet.source_pe) * cores_per_pe_ + packet.source_core;
}

bool EpochCoordinatorV5::allReports_() const {
    return std::all_of(seen_.begin(), seen_.end(), [](bool value) { return value; });
}

void EpochCoordinatorV5::resetReports_() {
    std::fill(seen_.begin(), seen_.end(), false);
    last_progress_cycle_ = cycle_;
    phase_start_ns_ = getCurrentSimTimeNano();
}

void EpochCoordinatorV5::sendAll_(CoreControlOp operation, std::uint64_t epoch) {
    for (std::uint32_t pe = 0; pe < pes_; ++pe) {
        for (std::uint32_t core = 0; core < cores_per_pe_; ++core) {
            auto* packet = new NocControlV5Event();
            packet->kind = NocControlV5Kind::Command;
            packet->operation = operation;
            packet->epoch = epoch;
            packet->source_pe = 0;
            packet->source_core = 0;
            packet->destination_pe = pe;
            packet->destination_core = core;
            command_->send(packet);
            ++commands_;
        }
    }
}

void EpochCoordinatorV5::advance_(CoreControlOp operation) {
    const auto now = std::uint64_t(getCurrentSimTimeNano());
    if (now >= phase_start_ns_) barrier_wait_ns_ += now - phase_start_ns_;
    sendAll_(operation, epoch_);
    resetReports_();
}

void EpochCoordinatorV5::handleStatus_(SST::Event* event) {
    auto* packet = dynamic_cast<NocControlV5Event*>(event);
    if (!packet || packet->kind != NocControlV5Kind::Status) {
        delete event; out_.fatal(CALL_INFO, -1, "EpochCoordinatorV5 received an invalid status\n");
    }
    ++reports_;
    last_progress_cycle_ = cycle_;

    if (packet->operation == CoreControlOp::IngressProgress) {
        if (packet->epoch != epoch_ || phase_ != Phase::Ingress) {
            delete packet; out_.fatal(CALL_INFO, -1, "cross-epoch ingress delivery report\n");
        }
        delivered_data_ += packet->logical_count;
        if (allReports_() && delivered_data_ == expected_data_) {
            phase_ = Phase::CommitReady;
            advance_(CoreControlOp::SealIngress);
        }
        delete packet;
        return;
    }

    const auto index = participant_(*packet);
    if (seen_[index]) {
        delete packet; out_.fatal(CALL_INFO, -1, "duplicate epoch report\n");
    }
    if (phase_ != Phase::Preload && packet->epoch != epoch_) {
        delete packet; out_.fatal(CALL_INFO, -1, "cross-epoch control report\n");
    }

    CoreControlOp expected = CoreControlOp::Abort;
    switch (phase_) {
    case Phase::Preload: expected = CoreControlOp::PreloadReady; break;
    case Phase::Ingress: expected = CoreControlOp::IngressReady; break;
    case Phase::CommitReady: expected = CoreControlOp::CommitReady; break;
    case Phase::CommitDone: expected = CoreControlOp::CommitDone; break;
    case Phase::Finished: break;
    }
    if (packet->operation != expected) {
        delete packet; out_.fatal(CALL_INFO, -1, "unexpected report for current epoch phase\n");
    }
    seen_[index] = true;
    if (phase_ == Phase::Ingress) expected_data_ += packet->logical_count;

    if (allReports_()) {
        if (phase_ == Phase::Preload) {
            phase_ = Phase::Ingress;
            expected_data_ = 0;
            delivered_data_ = 0;
            advance_(CoreControlOp::Start);
        } else if (phase_ == Phase::Ingress) {
            if (delivered_data_ == expected_data_) {
                phase_ = Phase::CommitReady;
                advance_(CoreControlOp::SealIngress);
            }
        } else if (phase_ == Phase::CommitReady) {
            phase_ = Phase::CommitDone;
            advance_(CoreControlOp::Commit);
        } else if (phase_ == Phase::CommitDone) {
            ++epochs_completed_;
            if (epochs_completed_ == timesteps_) {
                const auto now = std::uint64_t(getCurrentSimTimeNano());
                if (now >= phase_start_ns_) barrier_wait_ns_ += now - phase_start_ns_;
                phase_ = Phase::Finished;
                writeEvidence_();
                primaryComponentOKToEndSim();
            } else {
                ++epoch_;
                phase_ = Phase::Ingress;
                expected_data_ = 0;
                delivered_data_ = 0;
                advance_(CoreControlOp::Start);
            }
        }
    }
    delete packet;
}

bool EpochCoordinatorV5::tick_(SST::Cycle_t) {
    ++cycle_;
    if (phase_ != Phase::Finished && cycle_ - last_progress_cycle_ > timeout_cycles_) {
        ++timeouts_;
        sendAll_(CoreControlOp::Abort, epoch_);
        out_.fatal(CALL_INFO, -1, "P5 epoch control timeout at epoch=%llu phase=%u\n",
                   static_cast<unsigned long long>(epoch_), static_cast<unsigned>(phase_));
    }
    return false;
}

void EpochCoordinatorV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_);
    out << "{\n  \"protocol\": \"timed_control_vn\",\n"
        << "  \"data_vn\": " << kNocDataVn << ",\n"
        << "  \"control_vn\": " << kNocControlVn << ",\n"
        << "  \"participants\": " << seen_.size() << ",\n"
        << "  \"start_timestep\": " << start_timestep_ << ",\n"
        << "  \"timesteps\": " << timesteps_ << ",\n"
        << "  \"epochs_completed\": " << epochs_completed_ << ",\n"
        << "  \"commands\": " << commands_ << ",\n"
        << "  \"reports\": " << reports_ << ",\n"
        << "  \"barrier_wait_ns\": " << barrier_wait_ns_ << ",\n"
        << "  \"timeouts\": " << timeouts_ << ",\n"
        << "  \"status\": \"" << (phase_ == Phase::Finished ? "PASS" : "INCOMPLETE") << "\"\n}\n";
}

void EpochCoordinatorV5::finish() {
    if (phase_ != Phase::Finished) writeEvidence_();
    registerStatistic<std::uint64_t>("sync.commands")->addData(commands_);
    registerStatistic<std::uint64_t>("sync.reports")->addData(reports_);
    registerStatistic<std::uint64_t>("sync.epochs_completed")->addData(epochs_completed_);
    registerStatistic<std::uint64_t>("sync.barrier_wait_ns")->addData(barrier_wait_ns_);
    registerStatistic<std::uint64_t>("sync.timeouts")->addData(timeouts_);
}

}}}

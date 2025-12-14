#include "GasPhaseController.h"
#include "SnnPESubComponent.h"

using namespace SST::SnnDL;

void GasPhaseController::beginApplyFullSequence(bool strict_active) {
    if (!core_) return;
    core_->orchestratePrepareApplyWindow();
    core_->orchestrateApplyWindowEntry();
    core_->orchestrateBeginApplyIssueFallback(strict_active);
}

void GasPhaseController::beginScatterSequence() {
    if (!core_) return;
    core_->orchestrateBeginScatterSequence();
}

void GasPhaseController::endScatterSequence() {
    if (!core_) return;
    core_->orchestrateEndScatterSequence();
}


#include "header.hpp"

LearningRateDecayFuncBase::LearningRateDecayFuncBase(float initial_lr, float min_lr, int total_epoch)
    : initial_lr(initial_lr), min_lr(min_lr), total_epoch(total_epoch) {}

// ======================================================================================================================

StepDecay::StepDecay(float initial_lr, float min_lr, int decay_step_size, float decay_factor) : LearningRateDecayFuncBase(initial_lr, min_lr, 0), decay_step_size(decay_step_size), decay_factor(decay_factor) {}

void StepDecay::setDecayConstants(int dss, float df) {
    decay_step_size = dss;
    decay_factor = df;
}

float StepDecay::decay(int timestep) {
    return max(min_lr, initial_lr * pow(decay_factor, floor(static_cast<float>(timestep)/decay_step_size)));
}

// ======================================================================================================================

ExponentialDecay::ExponentialDecay(float initial_lr, float min_lr, float decay_constant) : LearningRateDecayFuncBase(initial_lr, min_lr, 0), decay_constant(decay_constant) {}

void ExponentialDecay::setDecayConstant(float dc) {
    decay_constant = dc;
}

float ExponentialDecay::decay(int timestep) {
    return max(min_lr, initial_lr * exp(-decay_constant*timestep));
}

// ======================================================================================================================

LinearDecay::LinearDecay(float initial_lr, float min_lr, int total_epoch) : LearningRateDecayFuncBase(initial_lr, min_lr, total_epoch) {}

void LinearDecay::setTotalEpoch(int T) {
    total_epoch = T;
}

float LinearDecay::decay(int timestep) {
    return max(min_lr, initial_lr * (1.0f - (static_cast<float>(timestep) / total_epoch)));
}

// ======================================================================================================================

CosineAnnealing::CosineAnnealing(float initial_lr, float min_lr, int total_epoch) : LearningRateDecayFuncBase(initial_lr, min_lr, total_epoch) {}

void CosineAnnealing::setTotalEpoch(int T) {
    total_epoch = T;
}

float CosineAnnealing::decay(int timestep) {
    int t = min(timestep, total_epoch);
    float cosine = cos(M_PI * t / total_epoch);
    return min_lr + 0.5f * (initial_lr - min_lr) * (1.0f + cosine);
}
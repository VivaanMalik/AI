#include "header.hpp"

// OptimizingFuncBase::OptimizingFuncBase() {}

void print_weights(float* d_weights, int total_size) {
    vector<float> h_weights(total_size);
    
    cudaMemcpy(h_weights.data(), d_weights, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << "Weights: ";
    for (int i = 0; i < 10; ++i) {
        cout << h_weights[i] << " ";
    }
    cout << endl;
}

StochasticGradientDescent::StochasticGradientDescent(float LR) : lr(LR), PNC(0), NC(0) {}
StochasticGradientDescent::~StochasticGradientDescent() {}

__global__ void SGD_step_kernel(float* weights_or_biases, float* dW_ordB, int size_of_thing, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<size_of_thing) {
        weights_or_biases[idx] -= lr * dW_ordB[idx];
    }
}

void StochasticGradientDescent::step(float* weights, float* biases, float* dW, float* dB) {
    int total_size = PNC * NC;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    SGD_step_kernel<<<blocks, threads>>>(weights, dW, total_size, lr);
    
    blocks = (NC + threads - 1) / threads;
    SGD_step_kernel<<<blocks, threads>>>(biases, dB, NC, lr);

    checkError("Normal SGD shit");
}

void StochasticGradientDescent::SetSize(int PrevNodeCount, int NodeCount) {
    PNC = PrevNodeCount;
    NC = NodeCount;
}

void StochasticGradientDescent::SetNewLR(float NewLR) {
    lr = NewLR;
}

// ======================================================================================================================

SGDMomentum::SGDMomentum(float momentum_coeff, float LR) : MomentumCoeff(momentum_coeff), lr(LR), WeightVel(nullptr), BiasVel(nullptr), PNC(0), NC(0) {}
SGDMomentum::~SGDMomentum() {
    if (WeightVel) cudaFree(WeightVel);
    if (BiasVel) cudaFree(BiasVel);
}

__global__ void SGDMomentum_step_kernel(float* weights_or_biases, float* dW_ordB, float* Vel, int size_of_thing, float lr, float MomentumCoeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size_of_thing) {
        Vel[idx] = MomentumCoeff * Vel[idx] - lr * dW_ordB[idx];
        weights_or_biases[idx] += Vel[idx];

        // weights_or_biases[idx] -= lr * dW_ordB[idx];
    }
}

void SGDMomentum::step(float* weights, float* biases, float* dW, float* dB) {
    int total_size = PNC * NC;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    SGDMomentum_step_kernel<<<blocks, threads>>>(weights, dW, WeightVel, total_size, lr, MomentumCoeff);

    checkError("Weight shit");
    
    blocks = (NC + threads - 1) / threads;
    SGDMomentum_step_kernel<<<blocks, threads>>>(biases, dB, BiasVel, NC, lr, MomentumCoeff);

    checkError("Bias shit");
}

void SGDMomentum::SetSize(int PrevNodeCount, int NodeCount) {
    PNC = PrevNodeCount;
    NC = NodeCount;

    if (WeightVel) cudaFree(WeightVel);
    if (BiasVel) cudaFree(BiasVel);

    cudaMalloc(&WeightVel, PNC * NC * sizeof(float));
    cudaMalloc(&BiasVel, NC * sizeof(float));

    cudaMemset(WeightVel, 0, PNC * NC * sizeof(float));
    cudaMemset(BiasVel, 0, NC * sizeof(float));
    checkError("Defining shit");
}

void SGDMomentum::SetNewLR(float NewLR) {
    lr = NewLR;
}

// ======================================================================================================================

NesterovAcceleratedGradient::NesterovAcceleratedGradient(float momentum_coeff, float LR) : MomentumCoeff(momentum_coeff), lr(LR), WeightVel(nullptr), BiasVel(nullptr), PNC(0), NC(0) {}
NesterovAcceleratedGradient::~NesterovAcceleratedGradient() {
    if (WeightVel) cudaFree(WeightVel);
    if (BiasVel) cudaFree(BiasVel);
}

__global__ void NesterovAcceleratedGradient_step_kernel(float* weights_or_biases, float* dW_ordB, float* Vel, int size_of_thing, float lr, float MomentumCoeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size_of_thing) {
        Vel[idx] = MomentumCoeff * Vel[idx] - lr * dW_ordB[idx];
        weights_or_biases[idx] += Vel[idx];

        // weights_or_biases[idx] -= lr * dW_ordB[idx];
    }
}

void NesterovAcceleratedGradient::step(float* weights, float* biases, float* dW, float* dB) {
    int total_size = PNC * NC;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    NesterovAcceleratedGradient_step_kernel<<<blocks, threads>>>(weights, dW, WeightVel, total_size, lr, MomentumCoeff);

    checkError("Weight shit");
    
    blocks = (NC + threads - 1) / threads;
    NesterovAcceleratedGradient_step_kernel<<<blocks, threads>>>(biases, dB, BiasVel, NC, lr, MomentumCoeff);

    checkError("Bias shit");
}

__global__ void NesterovAcceleratedGradient_temp_kernel(float* weights_or_biases, float* Vel, int size_of_thing, float MomentumCoeff, int positive_negtive) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size_of_thing) {
        weights_or_biases[idx] += positive_negtive * MomentumCoeff * Vel[idx];
    }
}

void NesterovAcceleratedGradient::TemporaryUpdate(float* weights, float* biases, int positive_negative /* 1, -1*/) {
    int total_size = PNC * NC;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    NesterovAcceleratedGradient_temp_kernel<<<blocks, threads>>>(weights, WeightVel, total_size, MomentumCoeff, positive_negative);
    checkError("Weight shit");
    blocks = (NC + threads - 1) / threads;
    NesterovAcceleratedGradient_temp_kernel<<<blocks, threads>>>(biases, BiasVel, NC, MomentumCoeff, positive_negative);
    checkError("Bias shit");
}

void NesterovAcceleratedGradient::SetSize(int PrevNodeCount, int NodeCount) {
    PNC = PrevNodeCount;
    NC = NodeCount;

    if (WeightVel) cudaFree(WeightVel);
    if (BiasVel) cudaFree(BiasVel);

    cudaMalloc(&WeightVel, PNC * NC * sizeof(float));
    cudaMalloc(&BiasVel, NC * sizeof(float));

    cudaMemset(WeightVel, 0, PNC * NC * sizeof(float));
    cudaMemset(BiasVel, 0, NC * sizeof(float));
    checkError("Defining shit");
}

void NesterovAcceleratedGradient::SetNewLR(float NewLR) {
    lr = NewLR;
}

// ======================================================================================================================

RMSProp::RMSProp(float decay_rate, float LR) : DecayRate(decay_rate), lr(LR), SqWeightGradAvg(nullptr), SqBiasGradAvg(nullptr), PNC(0), NC(0), Epsilon(1e-9f) {}
RMSProp::~RMSProp() {
    if (SqWeightGradAvg) cudaFree(SqWeightGradAvg);
    if (SqBiasGradAvg) cudaFree(SqBiasGradAvg);
}

__global__ void RMSProp_step_kernel(float* weights_or_biases, float* dW_ordB, float* SqGradAvg, int size_of_thing, float lr, float DecayRate, float Epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size_of_thing) {
        SqGradAvg[idx] = (DecayRate * SqGradAvg[idx]) + ((1.0f-DecayRate) * dW_ordB[idx] * dW_ordB[idx]);
        weights_or_biases[idx] -= lr * dW_ordB[idx] / sqrtf(SqGradAvg[idx] + Epsilon);
    }
}

void RMSProp::step(float* weights, float* biases, float* dW, float* dB) {
    int total_size = PNC * NC;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    RMSProp_step_kernel<<<blocks, threads>>>(weights, dW, SqWeightGradAvg, total_size, lr, DecayRate, Epsilon);

    checkError("Weight shit");
    
    blocks = (NC + threads - 1) / threads;
    RMSProp_step_kernel<<<blocks, threads>>>(biases, dB, SqBiasGradAvg, NC, lr, DecayRate, Epsilon);

    checkError("Bias shit");
}

void RMSProp::SetSize(int PrevNodeCount, int NodeCount) {
    PNC = PrevNodeCount;
    NC = NodeCount;

    if (SqWeightGradAvg) cudaFree(SqWeightGradAvg);
    if (SqBiasGradAvg) cudaFree(SqBiasGradAvg);

    cudaMalloc(&SqWeightGradAvg, PNC * NC * sizeof(float));
    cudaMalloc(&SqBiasGradAvg, NC * sizeof(float));

    cudaMemset(SqWeightGradAvg, 0, PNC * NC * sizeof(float));
    cudaMemset(SqBiasGradAvg, 0, NC * sizeof(float));
    checkError("Defining shit");
}

void RMSProp::SetNewLR(float NewLR) {
    lr = NewLR;
}

// ======================================================================================================================

Adam::Adam(float first_moment_decay_rate, float second_moment_decay_rate, float LR) : FirstMomentDecayRate(first_moment_decay_rate), 
  SecondMomentDecayRate(second_moment_decay_rate), lr(LR), FirstMomentWeight(nullptr), SecondMomentWeight(nullptr), 
  FirstMomentBias(nullptr), SecondMomentBias(nullptr), PNC(0), NC(0), Epsilon(1e-9f), TimeStep(0) {}
Adam::~Adam() {
    if (FirstMomentWeight) cudaFree(FirstMomentWeight);
    if (SecondMomentWeight) cudaFree(SecondMomentWeight);
    if (FirstMomentBias) cudaFree(FirstMomentBias);
    if (SecondMomentBias) cudaFree(SecondMomentBias);
}

__global__ void Adam_step_kernel(float* weights_or_biases, float* dW_ordB, float* FirstMomentThing, float* SecondMomentThing, int size_of_thing, float lr, float FirstMomentDecayRate, float SecondMomentDecayRate, float Epsilon, int TimeStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx<size_of_thing) {
        // first moment is mean
        // second moment is variance
        FirstMomentThing[idx] = FirstMomentThing[idx] * FirstMomentDecayRate + (1.0f - FirstMomentDecayRate) * dW_ordB[idx];
        SecondMomentThing[idx] = SecondMomentThing[idx] * SecondMomentDecayRate + (1.0f - SecondMomentDecayRate) * dW_ordB[idx] * dW_ordB[idx];
        float BiasCorrectedMeanThing = FirstMomentThing[idx] / (1.0f - (powf(FirstMomentDecayRate, TimeStep)));
        float BiasCorrectedVarianceThing = SecondMomentThing[idx] / (1.0f - (powf(SecondMomentDecayRate, TimeStep)));
        weights_or_biases[idx] -= lr * BiasCorrectedMeanThing / (sqrtf(BiasCorrectedVarianceThing) + Epsilon);
    }
}

void Adam::step(float* weights, float* biases, float* dW, float* dB) {
    TimeStep++;

    int total_size = PNC * NC;

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    Adam_step_kernel<<<blocks, threads>>>(weights, dW, FirstMomentWeight, SecondMomentWeight, total_size, lr, FirstMomentDecayRate, SecondMomentDecayRate, Epsilon, TimeStep);

    checkError("Weight shit");
    
    blocks = (NC + threads - 1) / threads;
    Adam_step_kernel<<<blocks, threads>>>(biases, dB, FirstMomentBias, SecondMomentBias, NC, lr, FirstMomentDecayRate, SecondMomentDecayRate, Epsilon, TimeStep);

    checkError("Bias shit");
}

void Adam::SetSize(int PrevNodeCount, int NodeCount) {
    PNC = PrevNodeCount;
    NC = NodeCount;

    if (FirstMomentWeight) cudaFree(FirstMomentWeight);
    if (SecondMomentWeight) cudaFree(SecondMomentWeight);
    if (FirstMomentBias) cudaFree(FirstMomentBias);
    if (SecondMomentBias) cudaFree(SecondMomentBias);

    cudaMalloc(&FirstMomentWeight, PNC * NC * sizeof(float));
    cudaMalloc(&SecondMomentWeight, PNC * NC * sizeof(float));
    cudaMalloc(&FirstMomentBias, NC * sizeof(float));
    cudaMalloc(&SecondMomentBias, NC * sizeof(float));

    cudaMemset(FirstMomentWeight, 0, PNC * NC * sizeof(float));
    cudaMemset(SecondMomentWeight, 0, PNC * NC * sizeof(float));
    cudaMemset(FirstMomentBias, 0, NC * sizeof(float));
    cudaMemset(SecondMomentBias, 0, NC * sizeof(float));
    checkError("Defining shit");
}

void Adam::SetNewLR(float NewLR) {
    lr = NewLR;
}
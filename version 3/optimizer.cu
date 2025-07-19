#include "header.hpp"

// OptimizingFuncBase::OptimizingFuncBase() {}

StochasticGradientDescent::StochasticGradientDescent(float LR) : lr(LR) {}
StochasticGradientDescent::~StochasticGradientDescent() {}

void print_weights(float* d_weights, int total_size) {
    vector<float> h_weights(total_size);
    
    cudaMemcpy(h_weights.data(), d_weights, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Weights: ";
    for (int i = 0; i < 10; ++i) {
        cout << h_weights[i] << " ";
    }
    cout << endl;
}

__global__ void SGD_step_kernel_W(float* weights, float* dW, int total_size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<total_size) {
        weights[idx] -= lr * dW[idx];
    }
}

__global__ void SGD_step_kernel_B(float* biases, float* dB, int NodeCount, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<NodeCount) {
        biases[idx] -= lr * dB[idx];
    }
}

void StochasticGradientDescent::step(float* weights, float* biases, float* dW, float* dB, int PrevNodeCount, int NodeCount) {
    int total_size = PrevNodeCount * NodeCount;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    // cout<< "prestep: \n";
    // print_weights(weights, total_size);
    // cout << endl;
    SGD_step_kernel_W<<<blocks, threads>>>(weights, dW, total_size, lr);
    // cout<< "poststep: \n";
    // print_weights(weights, total_size);
    // cout << endl;
    // this_thread::sleep_for(std::chrono::seconds(1));
    
    blocks = (NodeCount + threads - 1) / threads;
    SGD_step_kernel_B<<<blocks, threads>>>(biases, dB, NodeCount, lr);
}

void StochasticGradientDescent::SetNewLR(float NewLR) {
    lr = NewLR;
}
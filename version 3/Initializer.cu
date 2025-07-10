#include "header.hpp"
#include <curand.h>
#include <curand_kernel.h>

InitializerBase::InitializerBase() {}


__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

Xavier::Xavier(): d_weights(nullptr) {}
Xavier::~Xavier() {
    if (d_weights) cudaFree(d_weights);
}

__global__ void xavier_intialize(curandState* state, float* d_weights, int batch_size, int feature_size, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float multiplier = sqrtf(2.0f/(batch_size+feature_size));
    if (idx < total_size) {
        d_weights[idx] = curand_uniform(&state[idx]) * multiplier;
    }
}

float* Xavier::initialize(int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;
    
    cudaMalloc(&d_weights, total_size * sizeof(float));

    // set random
    curandState *d_state;
    cudaMalloc(&d_state, total_size * sizeof(curandState));

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, time(NULL));
    xavier_intialize<<<num_blocks, threads_per_block>>>(d_state, d_weights, batch_size, feature_size, total_size);

    cudaFree(d_state);

    return d_weights;
}
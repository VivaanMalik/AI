#include "header.hpp"
#include <curand.h>
#include <curand_kernel.h>

InitializerBase::InitializerBase() {}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, idx, &state[idx]);
}

__global__ void check_rng(curandState* state, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float r1 = curand_uniform(&state[idx]);
        float r2 = curand_normal(&state[idx]);
        printf("Thread %d → uniform: %f, normal: %f\n", idx, r1, r2);
    }
}

XavierNormal::XavierNormal(): d_weights(nullptr) {}
XavierNormal::~XavierNormal() {
    if (d_weights) cudaFree(d_weights);
}

__global__ void xavier_normal_initialize(curandState* state, float* d_weights, float multiplier, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        d_weights[idx] = curand_normal(&state[idx]) * multiplier;
    }
}

float* XavierNormal::initialize(int shape_0, int shape_1) {
    int total_size = shape_0 * shape_1;
    float multiplier = sqrtf(2.0f / (shape_0 + shape_1));
    
    cudaMalloc(&d_weights, total_size * sizeof(float));

    // set random
    curandState *d_state;
    cudaMalloc(&d_state, total_size * sizeof(curandState));
    cudaMemset(d_weights, 0, total_size * sizeof(float));
    cudaMemset(d_state, 0, total_size * sizeof(curandState));

    auto now = chrono::high_resolution_clock::now();
    auto seed = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, seed);
    cudaDeviceSynchronize();
    xavier_normal_initialize<<<num_blocks, threads_per_block>>>(d_state, d_weights, multiplier, total_size);
    cudaDeviceSynchronize();

    cudaFree(d_state);

    return d_weights;
}

// ======================================================================================================================

XavierUniform::XavierUniform(): d_weights(nullptr) {}
XavierUniform::~XavierUniform() {
    if (d_weights) cudaFree(d_weights);
}

__global__ void xavier_uniform_initialize(curandState* state, float* d_weights, float multiplier, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        d_weights[idx] = 2*(curand_uniform(&state[idx])-0.5f) * multiplier;
    }
}

float* XavierUniform::initialize(int shape_0, int shape_1) {
    int total_size = shape_0 * shape_1;
    float multiplier = sqrtf(6.0f/(shape_0+shape_1));
    
    cudaMalloc(&d_weights, total_size * sizeof(float));

    // set random
    curandState *d_state;
    cudaMalloc(&d_state, total_size * sizeof(curandState));
    cudaMemset(d_weights, 0, total_size * sizeof(float));
    cudaMemset(d_state, 0, total_size * sizeof(curandState));

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, seed);
    cudaDeviceSynchronize();
    xavier_uniform_initialize<<<num_blocks, threads_per_block>>>(d_state, d_weights, multiplier, total_size);
    cudaDeviceSynchronize();

    cudaFree(d_state);

    return d_weights;
}

// ======================================================================================================================

HeUniform::HeUniform(): d_weights(nullptr) {}
HeUniform::~HeUniform() {
    if (d_weights) cudaFree(d_weights);
}

__global__ void he_uniform_initialize(curandState* state, float* d_weights, float multiplier, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        d_weights[idx] = 2*(curand_uniform(&state[idx])-0.5f) * multiplier;
    }
}

float* HeUniform::initialize(int shape_0, int shape_1) {
    int total_size = shape_0 * shape_1;
    float multiplier = sqrtf(6.0f/(shape_0));
    
    cudaMalloc(&d_weights, total_size * sizeof(float));

    // set random
    curandState *d_state;
    cudaMalloc(&d_state, total_size * sizeof(curandState));
    cudaMemset(d_weights, 0, total_size * sizeof(float));
    cudaMemset(d_state, 0, total_size * sizeof(curandState));

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, seed);
    cudaDeviceSynchronize();
    he_uniform_initialize<<<num_blocks, threads_per_block>>>(d_state, d_weights, multiplier, total_size);
    cudaDeviceSynchronize();

    cudaFree(d_state);

    return d_weights;
}

// ======================================================================================================================

HeNormal::HeNormal(): d_weights(nullptr) {}
HeNormal::~HeNormal() {
    if (d_weights) cudaFree(d_weights);
}

__global__ void he_normal_initialize(curandState* state, float* d_weights, float multiplier, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        d_weights[idx] = curand_normal(&state[idx]) * multiplier;
    }
}

float* HeNormal::initialize(int shape_0, int shape_1) {
    int total_size = shape_0 * shape_1;
    float multiplier = sqrtf(2.0f/(shape_0));
    
    cudaMalloc(&d_weights, total_size * sizeof(float));

    // set random
    curandState *d_state;
    cudaMalloc(&d_state, total_size * sizeof(curandState));
    cudaMemset(d_weights, 0, total_size * sizeof(float));
    cudaMemset(d_state, 0, total_size * sizeof(curandState));

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, seed);
    cudaDeviceSynchronize();
    he_normal_initialize<<<num_blocks, threads_per_block>>>(d_state, d_weights, multiplier, total_size);
    cudaDeviceSynchronize();

    cudaFree(d_state);

    return d_weights;
}
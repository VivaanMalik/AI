#include "utils.cuh"
#include <iostream>

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void check_rng(curandState* state, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float r1 = curand_uniform(&state[idx]);
        float r2 = curand_normal(&state[idx]);
        printf("Thread %d → uniform: %f, normal: %f\n", idx, r1, r2);
    }
}
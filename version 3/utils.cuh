#ifndef UTILS_CUH
#define UTILS_CUH

#include <curand_kernel.h>

__global__ void setup_kernel(curandState* state, unsigned long seed);
__global__ void check_rng(curandState* state, int size);

#endif
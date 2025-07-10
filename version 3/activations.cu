#include "header.hpp"

ActivationFuncBase::ActivationFuncBase() {}

Sigmoid::Sigmoid() : d_output(nullptr), d_backward_result(nullptr), current_size(0) {}
Sigmoid::~Sigmoid() {
    if (d_output) cudaFree(d_output);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
float* Sigmoid::forward(float* d_input, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_size);
    // cudaDeviceSynchronize();

    return d_output;
}
__global__ void sigmoid_backward_kernel(const float* grad,  const float* output, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = grad[idx] * output[idx] * (1.0f - output[idx]);
    }
}
float* Sigmoid::backward(float* d_grad, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_backward_result) cudaFree(d_backward_result);
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = total_size; 
    }

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size);
    // cudaDeviceSynchronize();

    return d_backward_result;
}

// ======================================================================================================================

Tanh::Tanh() : d_output(nullptr), d_backward_result(nullptr), current_size(0) {}
Tanh::~Tanh() {
    if (d_output) cudaFree(d_output);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void tanh_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}
float* Tanh::forward(float* d_input, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    tanh_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_size);
    // cudaDeviceSynchronize();

    return d_output;
}
__global__ void tanh_backward_kernel(const float* grad,  const float* output, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = grad[idx] * (1.0f - output[idx]*output[idx]);
    }
}
float* Tanh::backward(float* d_grad, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_backward_result) cudaFree(d_backward_result);
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = total_size; 
    }

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    tanh_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size);
    // cudaDeviceSynchronize();

    return d_backward_result;
}

// ======================================================================================================================

ReLU::ReLU() : d_output(nullptr), d_backward_result(nullptr), current_size(0) {}
ReLU::~ReLU() {
    if (d_output) cudaFree(d_output);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * (input[idx] > 0.0f);
    }
}
float* ReLU::forward(float* d_input, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_size);
    // cudaDeviceSynchronize();

    return d_output;
}
__global__ void relu_backward_kernel(const float* grad,  const float* output, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        result[idx] = (output[idx]>0.0f) ? grad[idx] : 0;
    }
}
float* ReLU::backward(float* d_grad, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_backward_result) cudaFree(d_backward_result);
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = total_size; 
    }

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size);
    // cudaDeviceSynchronize();

    return d_backward_result;
}

// ======================================================================================================================

LeakyReLU::LeakyReLU() : alpha(0.01), d_output(nullptr), d_backward_result(nullptr), current_size(0) {}
LeakyReLU::~LeakyReLU() {
    if (d_output) cudaFree(d_output);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void leaky_relu_kernel(const float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : alpha * input[idx];
    }
}
float* LeakyReLU::forward(float* d_input, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    leaky_relu_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_size, alpha);
    // cudaDeviceSynchronize();

    return d_output;
}
__global__ void leaky_relu_backward_kernel(const float* grad,  const float* output, float* result, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        result[idx] = (output[idx]>0.0f) ? grad[idx] : grad[idx] * alpha;
    }
}
float* LeakyReLU::backward(float* d_grad, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_backward_result) cudaFree(d_backward_result);
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = total_size; 
    }

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    leaky_relu_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size, alpha);
    // cudaDeviceSynchronize();

    return d_backward_result;
}
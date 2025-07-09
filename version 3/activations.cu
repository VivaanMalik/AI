#include "header.hpp"

ActivationFuncBase::ActivationFuncBase() {}

Sigmoid::Sigmoid() : d_input(nullptr), d_output(nullptr), current_size(0) {}
Sigmoid::~Sigmoid() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_grad) cudaFree(d_grad);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
float* Sigmoid::forward(vector<float>& pre_activation_values, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_input, total_size * sizeof(float));
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Copy to device
    cudaMemcpy(d_input, pre_activation_values.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

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
float* Sigmoid::backward(vector<float>& gradient, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_grad) cudaFree(d_grad);
        if (d_backward_result) cudaFree(d_backward_result);

        cudaMalloc(&d_grad, total_size * sizeof(float));
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = feature_size; 
    }

    // Copy inputs to device
    cudaMemcpy(d_grad, gradient.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size);
    // cudaDeviceSynchronize();

    return d_backward_result;
}

// ======================================================================================================================

ReLU::ReLU() : d_input(nullptr), d_output(nullptr), current_size(0) {}
ReLU::~ReLU() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_grad) cudaFree(d_grad);
    if (d_backward_result) cudaFree(d_backward_result);
}
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * (input[idx] > 0.0f);
    }
}
float* ReLU::forward(vector<float>& pre_activation_values, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_input, total_size * sizeof(float));
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Copy to device
    cudaMemcpy(d_input, pre_activation_values.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

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
float* ReLU::backward(vector<float>& gradient, int batch_size, int feature_size) {
    int total_size = batch_size * feature_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || feature_size != current_size) {
        if (d_grad) cudaFree(d_grad);
        if (d_backward_result) cudaFree(d_backward_result);

        cudaMalloc(&d_grad, total_size * sizeof(float));
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = feature_size; 
    }

    // Copy inputs to device
    cudaMemcpy(d_grad, gradient.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, total_size);
    // cudaDeviceSynchronize();

    return d_backward_result;
}
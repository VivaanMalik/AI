#include "header.hpp"

ActivationFuncBase::ActivationFuncBase() {}

Sigmoid::Sigmoid() : output(), d_input(nullptr), d_output(nullptr), current_size(0) {}

Sigmoid::~Sigmoid() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_grad) cudaFree(d_grad);
    if (d_backward_result) cudaFree(d_backward_result);
}

__global__ void sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

vector<vector<float>> Sigmoid::forward(vector<vector<float>>& pre_activation_values) {
    int batch_size = pre_activation_values.size();
    int feature_size = pre_activation_values[0].size();
    int total_size = batch_size * feature_size;

    vector<float> flat_input(total_size);
    flat_input = flatten(pre_activation_values);

    // Allocate GPU memory if size changed
    if (total_size > current_size) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_input, total_size * sizeof(float));
        cudaMalloc(&d_output, total_size * sizeof(float));
        current_size = total_size;
    }

    // Copy to device
    cudaMemcpy(d_input, flat_input.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, total_size);
    cudaDeviceSynchronize();

    // Copy result back
    vector<float> flat_output(total_size);
    cudaMemcpy(flat_output.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape to 2D
    vector<vector<float>> result(batch_size, vector<float>(feature_size));
    result = unflatten(flat_output, batch_size, feature_size);

    // Store 1D output for backward()
    output = move(flat_output);

    return result;
}

__global__ void sigmoid_backward_kernel(const float* grad,  const float* output, float* result, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * output_size;

    if (idx < total_size) {
        int j = idx % output_size;
        result[idx] = grad[idx] * output[j] * (1.0f - output[j]);
    }
}

vector<vector<float>> Sigmoid::backward(vector<vector<float>>& gradient) {
    int batch_size = gradient.size();
    int output_size = gradient[0].size();
    int total_size = batch_size * output_size;

    // Resize GPU buffers only if needed
    if (batch_size != last_batch_size || output_size != current_size) {
        if (d_grad) cudaFree(d_grad);
        if (d_backward_result) cudaFree(d_backward_result);

        cudaMalloc(&d_grad, total_size * sizeof(float));
        cudaMalloc(&d_backward_result, total_size * sizeof(float));
        last_batch_size = batch_size;
        current_size = output_size; 
    }

    // Flatten gradient to 1D (2D gay)
    vector<float> flat_grad(total_size);
    flat_grad = flatten(gradient);

    // Copy inputs to device
    cudaMemcpy(d_grad, flat_grad.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);  // already allocated in forward

    // Launch kernel
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(d_grad, d_output, d_backward_result, batch_size, output_size);
    cudaDeviceSynchronize();

    // Copy result back
    vector<float> flat_result(total_size);
    cudaMemcpy(flat_result.data(), d_backward_result, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Reshape to 2D
    vector<vector<float>> result(batch_size, vector<float>(output_size));
    result = unflatten(flat_result, batch_size, output_size);

    return result;
}
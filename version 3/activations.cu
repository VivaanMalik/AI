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

    // Flatten the 2D input
    vector<float> flat_input(total_size);
    for (int i = 0; i < batch_size; ++i)
        for (int j = 0; j < feature_size; ++j)
            flat_input[i * feature_size + j] = pre_activation_values[i][j];

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
    for (int i = 0; i < batch_size; ++i)
        for (int j = 0; j < feature_size; ++j)
            result[i][j] = flat_output[i * feature_size + j];

    // Store 1D output for backward()
    output = std::move(flat_output);

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
    for (int i = 0; i < batch_size; ++i)
        for (int j = 0; j < output_size; ++j)
            flat_grad[i * output_size + j] = gradient[i][j];

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
    for (int i = 0; i < batch_size; ++i)
        for (int j = 0; j < output_size; ++j)
            result[i][j] = flat_result[i * output_size + j];

    return result;
}
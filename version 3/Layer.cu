#include "header.hpp"
#include "utils.cuh"

Layer::Layer(int id, int prev_node_count, int node_count, InitializerBase* initialization_function, 
            ActivationFuncBase* activation_function, OptimizingFuncBase* optimizer_function, float probability_dropout)
: ID(id), PrevNodeCount(prev_node_count), NodeCount(node_count), InitializationFunctionClass(initialization_function), 
  ActivationFunctionClass(activation_function), OptimizerFunctionClass(optimizer_function), ProbabilityDropout(probability_dropout), 
  weights(nullptr), biases(nullptr), OutputValues(nullptr), dW(nullptr), dB(nullptr), dropout_mask(nullptr), input(nullptr), 
  d_state(nullptr), derivative_to_pass_on(nullptr) {
    int weight_size = PrevNodeCount * NodeCount;
    int bias_size = NodeCount;

    // Allocate weights, biases, gradients
    // weights dealt with in intitializer
    cudaMalloc(&biases, bias_size * sizeof(float));
    cudaMalloc(&dW, weight_size * sizeof(float));
    cudaMalloc(&dB, bias_size * sizeof(float));
}
Layer::~Layer() {
    if (biases) cudaFree(biases);
    if (dW) cudaFree(dW);
    if (dB) cudaFree(dB);
    if (derivative_to_pass_on) cudaFree(derivative_to_pass_on);
    if (OutputValues) cudaFree(OutputValues);
    if (dropout_mask) cudaFree(dropout_mask);
    if (d_state) cudaFree(d_state);
}

void Layer::initialize(int batch_size) {
    BatchSize = batch_size;
    input_size = BatchSize * PrevNodeCount;
    output_size = BatchSize * NodeCount;

    
    if (ProbabilityDropout > 0.0f) {
        cudaMalloc(&dropout_mask, output_size * sizeof(float));
    }
    cudaMalloc(&derivative_to_pass_on, input_size * sizeof(float));
    cudaMalloc(&OutputValues, output_size * sizeof(float));

    weights = InitializationFunctionClass->initialize(PrevNodeCount, NodeCount);
    cudaMemset(biases, 0, NodeCount * sizeof(float));

    cudaMalloc(&d_state, output_size * sizeof(curandState));
    cudaMemset(d_state, 0, output_size * sizeof(curandState));

    auto now = chrono::high_resolution_clock::now();
    auto seed = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();

    int threads_per_block = 256;
    int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_state, seed);
    cudaDeviceSynchronize();
}

__global__ void preactivation_calculation_kernel(float* input, float* weights, float* biases, float* output, int PrevNodeCount, 
                                                int NodeCount, int BatchSize) {

    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron < NodeCount && sample < BatchSize) {
        float sum = 0.0f;
        for (int i = 0; i < PrevNodeCount; i++) {
            float input_val = input[sample * PrevNodeCount + i];
            float weight_val = weights[i * NodeCount + neuron];
            sum += input_val * weight_val;
        }
        output[sample * NodeCount + neuron] = sum + biases[neuron];
    }
}

__global__ void dropout_kernel(float* values, float* mask, float prob, int output_size, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float randval = curand_uniform(&state[idx]);
        mask[idx] = (randval > prob) ? 1.0f : 0.0f;
        values[idx] *= mask[idx];
        values[idx] /= (1.0f - prob);
    }
}

void Layer::forward(float* inputvals) {
    input = inputvals;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (NodeCount + 15) / 16,
        (BatchSize + 15) / 16
    );

    preactivation_calculation_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        inputvals, weights, biases, OutputValues,
        PrevNodeCount, NodeCount, BatchSize
    );

    if (ActivationFunctionClass)
        ActivationFunctionClass->forward(OutputValues, BatchSize, NodeCount);

    // Dropout
    if (ProbabilityDropout > 0.0f) {
        int threads = 256;
        int blocks = (output_size + threads - 1) / threads;

        dropout_kernel<<<blocks, threads>>>(OutputValues, dropout_mask, ProbabilityDropout, output_size, d_state);
    }
}

__global__ void adjust_grad_kernel(float* values, float* mask, float prob, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        values[idx] *= mask[idx];
        values[idx] /= (1.0f - prob);
    }
}

__global__ void calculate_dW(float* input, float* gradients, float* dW, int PrevNodeCount, int NodeCount, int BatchSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev node index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // this layer ka node ka index

    if (i < PrevNodeCount && j < NodeCount) {
        float sum = 0.0f;
        for (int b = 0; b < BatchSize; b++) {
            float input_val = input[b * PrevNodeCount + i];
            float grad_val = gradients[b * NodeCount + j];
            sum += input_val * grad_val;
        }
        dW[i * NodeCount + j] = sum;
    }
}

__global__ void calculate_dB(float* gradients, float* dB, int BatchSize, int NodeCount) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron < NodeCount) {
        float sum = 0.0f;
        for (int b = 0; b < BatchSize; b++) {
            sum += gradients[b * NodeCount + neuron];
        }
        dB[neuron] = sum;
    }
}

__global__ void calculate_derivative_to_pass_on(float* weights, float* gradients, float* derivative_to_pass_on, int PrevNodeCount, int NodeCount, int BatchSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // prev node index
    int b = blockIdx.y * blockDim.y + threadIdx.y; // this layer ka node ka index

    if (i < PrevNodeCount && b < BatchSize) {
        float sum = 0.0f;
        for (int j = 0; j < NodeCount; j++) {
            float grad_val = gradients[b * NodeCount + j];
            float weight_val = weights[i * NodeCount + j];
            sum += grad_val * weight_val;
        }
        derivative_to_pass_on[b * PrevNodeCount + i] = sum;
    }
}

float* Layer::backward(float* grad_output) {
    if (ActivationFunctionClass)
        ActivationFunctionClass->backward(grad_output, BatchSize, NodeCount);

    if (ProbabilityDropout > 0.0f) {
        int threads = 256;
        int blocks = (output_size + threads - 1) / threads;

        adjust_grad_kernel<<<blocks, threads>>>(grad_output, dropout_mask, ProbabilityDropout, output_size);
    }

    dim3 threadsPerBlock_dW(16, 16);
    dim3 blocksPerGrid_dW(
        (PrevNodeCount + 15) / 16,
        (NodeCount + 15) / 16
    );
    calculate_dW<<<blocksPerGrid_dW, threadsPerBlock_dW>>>(input, grad_output, dW, PrevNodeCount, NodeCount, BatchSize);

    int threads = 256;
    int blocks = (NodeCount + threads - 1) / threads;
    calculate_dB<<<blocks, threads>>>(grad_output, dB, BatchSize, NodeCount);

    dim3 threadsPerBlock_derivative_to_pass_on(16, 16);
    dim3 blocksPerGrid_derivative_to_pass_on(
        (PrevNodeCount + 15) / 16,
        (BatchSize + 15) / 16
    );
    calculate_derivative_to_pass_on<<<blocksPerGrid_derivative_to_pass_on, threadsPerBlock_derivative_to_pass_on>>>(weights, grad_output, derivative_to_pass_on, PrevNodeCount, NodeCount, BatchSize);
    return derivative_to_pass_on;
}
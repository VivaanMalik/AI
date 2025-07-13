#include "header.hpp"

Layer::Layer(int id, int prev_node_count, int node_count, InitializerBase* initialization_function, ActivationFuncBase* activation_function, OptimizingFuncBase* optimizer_function, float probability_dropout)
: ID(id), PrevNodeCount(prev_node_count), NodeCount(node_count), InitializationFunctionClass(initialization_function), ActivationFunctionClass(activation_function), OptimizerFunctionClass(optimizer_function), ProbabilityDropout(probability_dropout) {
    int weight_size = PrevNodeCount * NodeCount;
    int bias_size = NodeCount;

    // Allocate weights, biases, gradients
    cudaMalloc(&biases, bias_size * sizeof(float));
    cudaMalloc(&dW, weight_size * sizeof(float));
    cudaMalloc(&dB, bias_size * sizeof(float));
}
Layer::~Layer() {
    if (biases) cudaFree(biases);
    if (dW) cudaFree(dW);
    if (dB) cudaFree(dB);

    if (input) cudaFree(input);

    if (PostActivationValues) cudaFree(PostActivationValues);

    if (dropout_mask) cudaFree(dropout_mask);
}

void Layer::initialize(int batch_size) {
    BatchSize = batch_size;
    int input_size = BatchSize * PrevNodeCount;
    int output_size = BatchSize * NodeCount;

    
    if (ProbabilityDropout > 0.0f) {
        cudaMalloc(&dropout_mask, output_size * sizeof(float));
    }

    cudaMalloc(&input, input_size * sizeof(float));
    cudaMalloc(&PostActivationValues, output_size * sizeof(float));

    weights = InitializationFunctionClass.initialize(PrevNodeCount, NodeCount);
    cudaMemset(biases, 0, NodeCount * sizeof(float));
}
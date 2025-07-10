#include "header.hpp"

RegularizationFuncBase::RegularizationFuncBase(float lambda_value) : lambda(lambda_value) {}

__device__ float sign(float x) {
    if (x > 0.0f) return 1.0f;
    else if (x < 0.0f) return -1.0f;
    else return 0.0f;
}

L1Regularization::L1Regularization(float lambda_value) : RegularizationFuncBase(lambda_value) {}
L1Regularization::~L1Regularization() {}

__global__ void l1regularization_loss_kernel(float* d_weights, float* d_loss, float lambda, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        atomicAdd(d_loss, lambda * fabsf(d_weights[idx]));
    }
}

void L1Regularization::UpdateLoss(float* d_weights, float* d_loss, int weight_size) {

    int threads_per_block = 256;
    int num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    l1regularization_loss_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_loss, this->lambda, weight_size);
    // cudaDeviceSynchronize();
}

__global__ void l1regularization_grad_kernel(float* d_weights, float* d_grad, float lambda, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        d_grad[idx] +=  lambda * sign(d_weights[idx]);
    }
}

void L1Regularization::UpdateGradient(float* d_weights, float* d_grad, int weight_size) {

    int threads_per_block = 256;
    int num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    l1regularization_grad_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_grad, this->lambda, weight_size);
    // cudaDeviceSynchronize();
}

// ======================================================================================================================

L2Regularization::L2Regularization(float lambda_value) : RegularizationFuncBase(lambda_value) {}
L2Regularization::~L2Regularization() {}

__global__ void l2regularization_loss_kernel(float* d_weights, float* d_loss, float lambda, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        atomicAdd(d_loss, lambda * d_weights[idx] * d_weights[idx]);
    }
}

void L2Regularization::UpdateLoss(float* d_weights, float* d_loss, int weight_size) {

    int threads_per_block = 256;
    int num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    l2regularization_loss_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_loss, this->lambda, weight_size);
    // cudaDeviceSynchronize();
}

__global__ void l2regularization_grad_kernel(float* d_weights, float* d_grad, float lambda, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        d_grad[idx] += 2 * lambda * d_weights[idx];
    }
}

void L2Regularization::UpdateGradient(float* d_weights, float* d_grad, int weight_size) {

    int threads_per_block = 256;
    int num_blocks = (weight_size + threads_per_block - 1) / threads_per_block;
    l2regularization_grad_kernel<<<num_blocks, threads_per_block>>>(d_weights, d_grad, this->lambda, weight_size);
    // cudaDeviceSynchronize();
}

// ======================================================================================================================
ElasticNet::ElasticNet(float lambda_value) : RegularizationFuncBase(lambda_value), l1reg(lambda_value), l2reg(lambda_value) {}
ElasticNet::~ElasticNet() {}

void ElasticNet::UpdateLoss(float* d_weights, float* d_loss, int weight_size) {
    l1reg.UpdateLoss(d_weights, d_loss, weight_size);
    l2reg.UpdateLoss(d_weights, d_loss, weight_size);
}

void ElasticNet::UpdateGradient(float* d_weights, float* d_grad, int weight_size) {
    l1reg.UpdateGradient(d_weights, d_grad, weight_size);
    l2reg.UpdateGradient(d_weights, d_grad, weight_size);
}
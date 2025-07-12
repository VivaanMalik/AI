#include "header.hpp"

// softmax + cce found here, not in activationss

LossFuncBase::LossFuncBase() {}

BinaryCrossEntropy::BinaryCrossEntropy() : predicted(nullptr), target(nullptr), grad(nullptr), outsize(-1), epsilon(1e-9f), current_size(0) {
    cudaMalloc(&d_loss, sizeof(float));
}
BinaryCrossEntropy::~BinaryCrossEntropy() {
    if (predicted) cudaFree(predicted);
    if (target) cudaFree(target);
    if (d_loss) cudaFree(d_loss);
}
__global__ void bce_loss_kernel(const float* y_pred, const float* y_true, float* d_loss, int total_elements, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], epsilon), 1.0f - epsilon); 
        atomicAdd(d_loss,- (y * logf(p) + (1.0f - y) * logf(1.0f - p)));
    }
}
float BinaryCrossEntropy::forward(float* d_pred, float* d_target, int batch_size, int num_classes) {
    int output_size = batch_size * num_classes; 

    // Allocate GPU memory if size changed
    if (output_size > current_size) {
        if (predicted) cudaFree(predicted);
        if (target) cudaFree(target);
        if (grad) cudaFree(grad);
        cudaMalloc(&predicted, output_size * sizeof(float));
        cudaMalloc(&target, output_size * sizeof(float));
        cudaMalloc(&grad, output_size * sizeof(float));
        current_size = output_size;
    }

    predicted = d_pred;
    target = d_target;
    outsize = output_size;

    cudaMemset(d_loss, 0, sizeof(float));

    int threads_per_block = 256;
    int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    bce_loss_kernel<<<num_blocks, threads_per_block>>>(predicted, target, d_loss, output_size, epsilon);
    // cudaDeviceSynchronize();

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss/output_size;
}
__global__ void bce_loss_backward_kernel(float* y_pred, float* y_true, float* grad, int total_elements, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float y = y_true[idx];
        float p = fminf(fmaxf(y_pred[idx], epsilon), 1.0f - epsilon); 
        grad[idx] = (p-y)/(p*(1-p)*total_elements);
    }
}
float* BinaryCrossEntropy::backward() {
    int threads_per_block = 256;
    int num_blocks = (outsize + threads_per_block - 1) / threads_per_block;
    bce_loss_backward_kernel<<<num_blocks, threads_per_block>>>(predicted, target, grad, outsize, epsilon);
    return grad;
}

// ======================================================================================================================

SoftmaxCategoricalCrossEntropy::SoftmaxCategoricalCrossEntropy() : probabilities(nullptr), target(nullptr), grad(nullptr), batchsize(-1), numclasses(-1), epsilon(1e-9f), current_size(0) {
    cudaMalloc(&d_loss, sizeof(float));
}
SoftmaxCategoricalCrossEntropy::~SoftmaxCategoricalCrossEntropy() {
    if (probabilities) cudaFree(probabilities);
    if (target) cudaFree(target);
    if (d_loss) cudaFree(d_loss);
}
__global__ void cce_loss_kernel(float* y_pred, float* y_true, float* probabilities, float* d_loss, int batch_size, int num_classes, float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size) {
        float* pred_row = y_pred + row * num_classes;
        float* target_row = y_true + row * num_classes;
        float* prob_row = probabilities + row * num_classes;

        // x_new = x - xp.max(x, axis=1, keepdims=True) # stable
        float max_val = pred_row[0];
        for (int j = 1; j < num_classes; ++j)
            max_val = fmaxf(max_val, pred_row[j]);

        // exp = xp.exp(x_new)
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            prob_row[j] = expf(pred_row[j] - max_val);
            sum_exp += prob_row[j];
        }

        // self.probs = exp / xp.sum(exp, axis=1, keepdims=True)
        for (int j = 0; j < num_classes; ++j)
            prob_row[j] /= sum_exp;

        // log_probs = -xp.log(xp.clip(xp.sum(self.probs * target, axis=1) + 1e-9, 1e-9, 1.0))
        float loss_i = 0.0f;
        for (int j = 0; j < num_classes; ++j)
            loss_i -= target_row[j] * logf(fmaxf(prob_row[j], epsilon));

        atomicAdd(d_loss, loss_i);
        }
}
float SoftmaxCategoricalCrossEntropy::forward(float* d_pred, float* d_target, int batch_size, int num_classes) {
    // Allocate GPU memory if size changed
    int output_size = batch_size * num_classes;
    if (output_size > current_size) {
        if (probabilities) cudaFree(probabilities);
        if (target) cudaFree(target);
        if (grad) cudaFree(grad);
        cudaMalloc(&probabilities, output_size * sizeof(float));
        cudaMalloc(&target, output_size * sizeof(float));
        cudaMalloc(&grad, output_size * sizeof(float));
        current_size = output_size;
    }

    target = d_target;
    batchsize = batch_size;
    numclasses = num_classes;

    cudaMemset(d_loss, 0, sizeof(float));

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    cce_loss_kernel<<<num_blocks, threads_per_block>>>(d_pred, target, probabilities, d_loss, batch_size, num_classes, epsilon);
    // cudaDeviceSynchronize();

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss/batch_size;
}
__global__ void cce_loss_backward_kernel(float* probabilities, float* y_true, float* grad, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        grad[idx] = (probabilities[idx] - y_true[idx]) / batch_size;
    }
}
float* SoftmaxCategoricalCrossEntropy::backward() {
    int threads_per_block = 256;
    int num_blocks = (batchsize*numclasses + threads_per_block - 1) / threads_per_block;
    cce_loss_backward_kernel<<<num_blocks, threads_per_block>>>(probabilities, target, grad, batchsize, numclasses);
    return grad;
}

// ======================================================================================================================

MeanSquaredLoss::MeanSquaredLoss() : predicted(nullptr), target(nullptr), grad(nullptr), outsize(-1), current_size(0) {
    cudaMalloc(&d_loss, sizeof(float));
}
MeanSquaredLoss::~MeanSquaredLoss() {
    if (predicted) cudaFree(predicted);
    if (target) cudaFree(target);
    if (d_loss) cudaFree(d_loss);
}
__global__ void mse_loss_kernel(const float* y_pred, const float* y_true, float* d_loss, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float y = y_true[idx];
        float p = y_pred[idx]; 
        atomicAdd(d_loss, (p-y) * (p-y));
    }
}
float MeanSquaredLoss::forward(float* d_pred, float* d_target, int batch_size, int num_classes) {
    int output_size = batch_size * num_classes; 

    // Allocate GPU memory if size changed
    if (output_size > current_size) {
        if (predicted) cudaFree(predicted);
        if (target) cudaFree(target);
        if (grad) cudaFree(grad);
        cudaMalloc(&predicted, output_size * sizeof(float));
        cudaMalloc(&target, output_size * sizeof(float));
        cudaMalloc(&grad, output_size * sizeof(float));
        current_size = output_size;
    }

    predicted = d_pred;
    target = d_target;
    outsize = output_size;

    cudaMemset(d_loss, 0, sizeof(float));

    int threads_per_block = 256;
    int num_blocks = (output_size + threads_per_block - 1) / threads_per_block;
    mse_loss_kernel<<<num_blocks, threads_per_block>>>(predicted, target, d_loss, output_size);
    // cudaDeviceSynchronize();

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss/output_size;
}
__global__ void mse_loss_backward_kernel(float* y_pred, float* y_true, float* grad, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float y = y_true[idx];
        float p = y_pred[idx]; 
        grad[idx] = 2.0f * (p-y);
    }
}
float* MeanSquaredLoss::backward() {
    int threads_per_block = 256;
    int num_blocks = (outsize + threads_per_block - 1) / threads_per_block;
    mse_loss_backward_kernel<<<num_blocks, threads_per_block>>>(predicted, target, grad, outsize);
    return grad;
}

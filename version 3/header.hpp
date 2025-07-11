#ifndef HEADER_HPP
#define HEADER_HPP

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <cmath> 
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

#define M_PI 3.14159265358979323846

using namespace std;

// Initializer
// ===============================================================================
class InitializerBase {
    public:
    InitializerBase();
    virtual float* initialize(int batch_size, int feature_size) = 0;
    virtual ~InitializerBase() {}
};

class XavierNormal : public InitializerBase {
public:
    float* d_weights;
    XavierNormal();
    ~XavierNormal();
    float* initialize(int batch_size, int feature_size);
};

class XavierUniform : public InitializerBase {
public:
    float* d_weights;
    XavierUniform();
    ~XavierUniform();
    float* initialize(int batch_size, int feature_size);
};

class HeUniform : public InitializerBase {
public:
    float* d_weights;
    HeUniform();
    ~HeUniform();
    float* initialize(int batch_size, int feature_size);
};

class HeNormal : public InitializerBase {
public:
    float* d_weights;
    HeNormal();
    ~HeNormal();
    float* initialize(int batch_size, int feature_size);
};

// ===============================================================================

// Activation
// ===============================================================================
class ActivationFuncBase {
    public:
    ActivationFuncBase();
    virtual float* forward(float* pre_activation_values, int batch_size, int feature_size) = 0;
    virtual float* backward(float* gradient, int batch_size, int feature_size) = 0;
    virtual ~ActivationFuncBase() {}
};

class Sigmoid : public ActivationFuncBase {
public:
    float* d_output;
    int current_size = 0;
    float* d_backward_result = nullptr;
    int last_batch_size = 0;

    Sigmoid();
    ~Sigmoid();

    float* forward(float* pre_activation_values, int batch_size, int feature_size);
    float* backward(float* gradient, int batch_size, int feature_size);
};

class Tanh : public ActivationFuncBase {
public:
    float* d_output;
    int current_size = 0;
    float* d_backward_result = nullptr;
    int last_batch_size = 0;

    Tanh();
    ~Tanh();

    float* forward(float* pre_activation_values, int batch_size, int feature_size);
    float* backward(float* gradient, int batch_size, int feature_size);
};

class ReLU : public ActivationFuncBase {
public:
    float* d_output;
    int current_size = 0;
    float* d_backward_result = nullptr;
    int last_batch_size = 0;

    ReLU();
    ~ReLU();

    float* forward(float* pre_activation_values, int batch_size, int feature_size);
    float* backward(float* gradient, int batch_size, int feature_size);
};

class LeakyReLU : public ActivationFuncBase {
public:
    float alpha = 0.01;
    float* d_output;
    int current_size = 0;
    float* d_backward_result = nullptr;
    int last_batch_size = 0;

    LeakyReLU();
    ~LeakyReLU();

    float* forward(float* pre_activation_values, int batch_size, int feature_size);
    float* backward(float* gradient, int batch_size, int feature_size);
};

// ===============================================================================

// Loss func
// ===============================================================================
class LossFuncBase {
    public:
    virtual float forward(vector<float> output, vector<float> target_output) = 0;
    virtual vector<float> backward() = 0;
    virtual ~LossFuncBase() {}
};
// ===============================================================================

// lr decay func
// ===============================================================================
class LearningRateDecayFuncBase {
    public:
    float initial_lr;
    float min_lr;
    int total_epoch;

    LearningRateDecayFuncBase(float initial_lr, float min_lr, int total_epoch = 0);
    virtual float decay(int timestep) = 0;
    virtual ~LearningRateDecayFuncBase() {}
};

class StepDecay : public LearningRateDecayFuncBase {
    public:
    int decay_step_size;
    float decay_factor;

    StepDecay(float initial_lr, float min_lr, int decay_step_size, float decay_factor = 0.5f);
    void setDecayConstants(int dss, float df = 0.5f);
    float decay(int timestep) override;
};

class ExponentialDecay : public LearningRateDecayFuncBase {
    public:
    float decay_constant;

    ExponentialDecay(float initial_lr, float min_lr, float decay_constant = 0.01f);
    void setDecayConstant(float dc);
    float decay(int timestep) override;
};

class LinearDecay : public LearningRateDecayFuncBase {
    public:
    LinearDecay(float initial_lr, float min_lr, int total_epoch = 0);
    void setTotalEpoch(int T);
    float decay(int timestep);
};

class CosineAnnealing : public LearningRateDecayFuncBase {
    public:
    CosineAnnealing(float initial_lr, float min_lr, int total_epoch = 0);
    void setTotalEpoch(int T);
    float decay(int timestep);
};
// ===============================================================================

// weight decay func (add loss)
// ===============================================================================
class RegularizationFuncBase {
    public:
    float lambda;
    explicit RegularizationFuncBase(float lambda);
    virtual void UpdateLoss(float* d_weights, float* d_loss, int weight_size) = 0;
    virtual void UpdateGradient(float* d_weights, float* d_grad, int weight_size) = 0;
    virtual ~RegularizationFuncBase() {}
};

class L1Regularization : public RegularizationFuncBase {
public:
    explicit L1Regularization(float lambda_value = 1e-4);
    ~L1Regularization();
    void UpdateLoss(float* d_weights, float* d_loss, int weight_size);
    void UpdateGradient(float* d_weights, float* d_grad, int weight_size);
};

class L2Regularization : public RegularizationFuncBase {
public:
    explicit L2Regularization(float lambda_value = 1e-4);
    ~L2Regularization();
    void UpdateLoss(float* d_weights, float* d_loss, int weight_size);
    void UpdateGradient(float* d_weights, float* d_grad, int weight_size);
};

class ElasticNet : public RegularizationFuncBase {
public:
    L1Regularization l1reg;
    L2Regularization l2reg;

    explicit ElasticNet(float lambda_value = 1e-4);
    ~ElasticNet();

    void UpdateLoss(float* d_weights, float* d_loss, int weight_size);
    void UpdateGradient(float* d_weights, float* d_grad, int weight_size);
};
// ===============================================================================

// Layer
// ===============================================================================
class Layer {
    public: 

};
// ===============================================================================


class Network {
    public:
    int id;
    vector<Layer> Layers;
    InitializerBase* Initializer;
    LossFuncBase* LossFunction;
    LearningRateDecayFuncBase* LearningRateDecayFunction;
    int EpochNumber;
    RegularizationFuncBase* RegularizationFunction;    
    Network(int id);

    void add_Layer(Layer layer);
    void log_work(chrono::steady_clock::time_point start);
    void test_activation_function();
    void test_initializer();
};

float GetElapsedTime(chrono::steady_clock::time_point);
string VectorFLoatToString(vector<float>);

string Print2DMatrix(vector<vector<float>>);
string Print1DVector(vector<float>);

vector<float> flatten(vector<vector<float>>);
vector<vector<float>> unflatten(vector<float>, int, int);

vector<float> to_cpu(const float*, size_t);
float* to_gpu(const vector<float>&);

#endif
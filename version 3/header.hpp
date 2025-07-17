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
#include <curand.h>
#include <curand_kernel.h>


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
    LossFuncBase();
    virtual float* forward(float* output, float* target_output, int batch_size, int num_classes) = 0;
    virtual float* backward() = 0;
    virtual ~LossFuncBase() {}
};

class BinaryCrossEntropy : public LossFuncBase {
    public: 
    float* predicted;
    float* target;
    int outsize;
    int current_size;
    float* d_loss;
    float* grad;

    BinaryCrossEntropy();
    ~BinaryCrossEntropy();

    float epsilon;
    float* forward(float* output, float* target_output, int batch_size, int num_classes);
    float* backward();
};

class SoftmaxCategoricalCrossEntropy : public LossFuncBase {
    public: 
    float* probabilities;
    float* target;
    int batchsize;
    int numclasses;
    int current_size;
    float* d_loss;
    float* grad;

    SoftmaxCategoricalCrossEntropy();
    ~SoftmaxCategoricalCrossEntropy();

    float epsilon;
    float* forward(float* output, float* target_output, int batch_size, int num_classes);
    float* backward();
};

class MeanSquaredLoss : public LossFuncBase {
    public: 
    float* predicted;
    float* target;
    int outsize;
    int current_size;
    float* d_loss;
    float* grad;

    MeanSquaredLoss();
    ~MeanSquaredLoss();

    float* forward(float* output, float* target_output, int batch_size, int num_classes);
    float* backward();
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

// Optimizer
// ===============================================================================
class OptimizingFuncBase {
    public:
    // OptimizingFuncBase();
    virtual void step(float* weights, float* biases, float* dW, float* dB, int PrevNodeCount, int NodeCount) = 0;
    virtual void SetNewLR(float NewLR) = 0;
    virtual ~OptimizingFuncBase() {}
};

class StochasticGradientDescent : public OptimizingFuncBase {
    public: 
    float lr;

    StochasticGradientDescent(float LR);
    ~StochasticGradientDescent();

    void step(float* weights, float* biases, float* dW, float* dB, int PrevNodeCount, int NodeCount);
    void SetNewLR(float NewLR);
};
// ===============================================================================

// Layer
// ===============================================================================
class Layer {
    public: 
    int ID; // mostly for debugging
    int BatchSize; // 
    int PrevNodeCount;
    int NodeCount;
    int input_size;
    int output_size;
    InitializerBase* InitializationFunctionClass;
    ActivationFuncBase* ActivationFunctionClass;
    OptimizingFuncBase* OptimizerFunctionClass;
    float* input;
    float* derivative_to_pass_on;
    float* OutputValues;
    float ProbabilityDropout = 0.0f;
    float* dropout_mask;
    float* weights;
    float* biases;
    float* dW;
    float* dB;
    curandState *d_state;

    Layer(int id, int prev_node_count, int node_count, InitializerBase* initialization_function, ActivationFuncBase* activation_function, OptimizingFuncBase* optimizer_function, float probability_dropout); // Primary assign: id and shit
    void initialize(int batch_size); // Secondary assign: weights and biases
    float* forward(float* inputvals);
    float* backward(float* grad_output);
    float* forward_prediction(float* inputval);
    ~Layer();
    
};
// ===============================================================================

class Network {
    public:
    int id;
    int BatchSize;
    int EpochNumber;
    vector<Layer*> Layers;
    LossFuncBase* LossFunction;
    LearningRateDecayFuncBase* LearningRateDecayFunction;
    RegularizationFuncBase* RegularizationFunction;    
    Network(int id);

    void add_Layer(Layer* layer);
    void compile_network(LossFuncBase* loss_func, LearningRateDecayFuncBase* Lrd_func, RegularizationFuncBase* reg_func, int batch_size);
    float* forward(float* x);
    void backward(float* grad);
    void train(vector<vector<float>> input_data, vector<vector<float>> output_target_data, int epoch);
    int predict(vector<float> input, int outputsize);
    float Evaluate(vector<vector<float>> input_data, vector<vector<float>> output_target_data);

    void log_work(chrono::steady_clock::time_point start);
    void test_activation_function();
    void test_initializer();
};

// wtf is this
// ===============================================================================
template<typename T, typename = void>
void setTotalEpochIfPossible(T* obj, int value) {
    // Do nothing if setTotalEpoch doesn't exist
}

// Overload when T has setTotalEpoch(int)
template<typename T>
auto setTotalEpochIfPossible(T* obj, int value)
    -> std::enable_if_t<
        std::is_same<decltype(std::declval<T>().setTotalEpoch(0)), void>::value
    >
{
    if (obj) obj->setTotalEpoch(value);
}
// ===============================================================================


float GetElapsedTime(chrono::steady_clock::time_point);
string VectorFLoatToString(vector<float>);

string Print2DMatrix(vector<vector<float>>);
string Print1DVector(vector<float>);

vector<float> flatten(vector<vector<float>>);
vector<vector<float>> unflatten(vector<float>, int, int);

vector<float> to_cpu(const float*, size_t);
float* to_gpu(const vector<float>&);

float round_to_sigfigs(float num, int n);

void log_location(const char* file, int line);
#define LOG_LOCATION() log_location(__FILE__, __LINE__)

void Program(int id);

#endif
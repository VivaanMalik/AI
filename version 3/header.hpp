#ifndef HEADER_HPP
#define HEADER_HPP

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include <cmath> 
#include <cuda_runtime.h>
#include "sqlite3/sqlite3.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
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
    virtual float forward(vector<float> output, vector<float> target_output) const;
    virtual vector<float> backward() const;
    virtual ~LossFuncBase() {}
};
// ===============================================================================

// lr decay func
// ===============================================================================
class LearningRateDecayFuncBase {
    public:
    LearningRateDecayFuncBase(float initial_lr);
    virtual float decay(int timestep) const;
    virtual ~LearningRateDecayFuncBase() {}
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
    void log_work(sqlite3* db_pointer, chrono::steady_clock::time_point start);
    void test_activation_function(std::string msg);
    void test_initializer();
};

sqlite3* setup_sqlite3_db();
void close_sqlite3_db(sqlite3*);
bool write_to_sqlite3_db(sqlite3*, string);
optional<json> read_from_sqlite3_db(sqlite3*, int);

json keepchecking(int, sqlite3*);
json ParseAndComputeData(string);
float GetElapsedTime(chrono::steady_clock::time_point);
string VectorFLoatToString(vector<float>);

string Print2DMatrix(vector<vector<float>>);
string Print1DVector(vector<float>);

vector<float> flatten(vector<vector<float>>);
vector<vector<float>> unflatten(vector<float>, int, int);

vector<float> to_cpu(const float*, size_t);
float* to_gpu(const vector<float>&);

#endif
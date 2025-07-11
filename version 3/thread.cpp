#include "header.hpp"

Network::Network(int id) : 
    id(id),
    Layers({}),
    Initializer(nullptr),
    LossFunction(nullptr),
    LearningRateDecayFunction(nullptr),
    EpochNumber(0),
    RegularizationFunction(nullptr) {
    }

void Network::add_Layer(Layer layer) {
    this->Layers.push_back(layer);
}

void Network::log_work(chrono::steady_clock::time_point start) {
    float created_on = GetElapsedTime(start);

    // this->test_activation_function();
    this->test_initializer();
}

void Network::test_activation_function() {
    // Sigmoid function;
    // ReLU function;
    LeakyReLU function;
    vector<float> v = { 0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f};

    float* d_v = to_gpu(v);
    int batch_size = 4; // vertical
    int feature_size = 4; // horizontal
    int total_size = batch_size * feature_size;
    
    float* result_pointer;

    result_pointer = function.forward(d_v, batch_size, feature_size);
    string actual_out = Print2DMatrix(unflatten(to_cpu(result_pointer, total_size), batch_size, feature_size));
    cout << actual_out + "\n";

    result_pointer = function.backward(d_v, batch_size, feature_size);
    actual_out = Print2DMatrix(unflatten(to_cpu(result_pointer, total_size), batch_size, feature_size));
    cout << actual_out + "\n";

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    result_pointer = function.forward(d_v, batch_size, feature_size);
    result_pointer = function.backward(d_v, batch_size, feature_size);
    float elapsed_time = GetElapsedTime(start);
    
    cout << "OUTPUT TIME: " + to_string(elapsed_time) + "\n";
}

void Network::test_initializer() {
    // HeUniform function;
    // XavierNormal function;
    // XavierUniform function;
    HeNormal function;

    int shape_0 = 4;
    int shape_1 = 8;
    float* d_weights;
    d_weights = function.initialize(shape_0, shape_1);
    vector<float> weights_vec = to_cpu(d_weights, shape_0*shape_1);
    string actual_out = Print2DMatrix(unflatten(weights_vec, shape_0, shape_1));
    cout << actual_out + "\n";

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    d_weights = function.initialize(shape_0, shape_1);
    float elapsed_time = GetElapsedTime(start);
    
    cout << "OUTPUT TIME: " + to_string(elapsed_time) + "\n";
}
# Unnamed Machine Learning Framework (temp name) [V3]
An ML framwork written in c++ and CUDA (only MLP as of now)

---

## Table of Contents
<!-- TOC -->
- [Unnamed Machine Learning Framework (temp name)](#unnamed-machine-learning-framework-temp-name)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
      - [Semisequential Setup](#semisequential-setup)
      - [Modifed Layers](#modifed-layers)
      - [Load data to and from a specified JSON file](#load-data-to-and-from-a-specified-json-file)
      - [Training function](#training-function)
      - [Evaluation function](#evaluation-function)
      - [Initialization Functions](#initialization-functions)
      - [Activation Functions](#activation-functions)
      - [Loss Functions](#loss-functions)
      - [Learning Rate Decay Functions](#learning-rate-decay-functions)
      - [Optimization Functions](#optimization-functions)
      - [Weight Decay Functions](#weight-decay-functions)
      - [Utility Functions](#utility-functions)
  - [Documentation of classes and methods (per file)](#documentation-of-classes-and-methods-per-file)
      - [network](#network)
      - [layer](#layer)
      - [initializer](#initializer)
      - [activations](#activations)
      - [losses](#losses)
      - [optimizer](#optimizer)
      - [learning\_rate\_decay](#learning_rate_decay)
      - [weight\_decay](#weight_decay)
  - [Example program](#example-program)


## Features
#### Semisequential Setup
> [!WARNING]
>**Incoming yap**

You can add layers to your network and compile them, but a few functions have to be defined as a parameter when the layer is defined.

The functions are not separated from the layer as I did not know about the "Sequential Setup" that is widely used until I tested my network on tensorflow to compare speeds and accuracy. 

This is a subtle flex that the first ML Framework I used (even semiseriously) was my own.

---

#### Modifed Layers
Each Layer has its own
* Unique ID
* Initialization function
* Activation function
* Optimization function
* Dropout probability
* Regularization function
  
---
  
#### Training function
* Auto shuffles datasets in batches
* Prints loss and time taken for every epoch

---

#### Evaluation function
Evaluates the network and send the following data
* Accuracy
* A label-wise confusion matrix that can be color coded using ```utils.cpp```

---

#### Initialization Functions
* Uniform Xavier
* Uniform He
* Normal Xavier
* Normal He

---

#### Activation Functions
* Sigmoid
* ReLU
* Leaky ReLU
* Softmax*

⚠️ **Important**: While Softmax is indeed an activation function, it has been defined within ```loss.py``` as a combination of Softmax and CCE since they are often used together.

---

#### Loss Functions
* Binary Cross Entropy (BCE)
* Mean Squared Error (MSE)
* Softmax + Categorical Cross Entropy (CCE)

---

#### Learning Rate Decay Functions
* Step Decay
* Exponential Decay
* Linear Decay
* Cosine Annealing

---

#### Optimization Functions
* Stochastic Gradient Descent (SGD)
* SGD with momentum
* Nesterov Accelerated Gradient (NAG)
* Root Mean Square Propogation (RMSProp)
* Adaptive Movement Estimation (Adam)

---

#### Weight Decay Functions
* L1 Regularization
* L2 Regularization
* ElasticNet Regularization

---

#### Some Utility Functions

* vector<float> to string
* vector<vector<float>> to string (with a cap and without)
* flatten vector<vector<float>>
* unflatten vector<float>
  
## Documentation of classes and methods (per file)

#### network
> Network(int id)
> ```
> id: Unique Network ID
> ```

> void add_Layer(Layer* layer)
> ```
> layer: Layer pointer to add to vector<Layer*> Layers
> ```

> void compile_network(LossFuncBase* loss_func, LearningRateDecayFuncBase* lrd_func, int batch_size)
> ```
> loss_func: Set loss function
> lrd_func: Set decay function for learning rate
> batch_size: Set Batch size
> ```

> void train(vector<vector<float>> input_data, vector<vector<float>> output_target_data, int epoch)
> ```
> input_data: Input value dataset
> output_target_data: Target value dataset
> epoch: No. of forward+baclward propogation cycles
> ```

> int predict(vector<float> input, int outputsize)
> ```
> input: Input value dataset list
> outputsize: No. of classes
> ```

> float Evaluate(vector<vector<float>> input_data, vector<vector<float>> output_target_data)
> ```
> input_data: Input value dataset list
> output_target_data: Target value dataset list
> ```

---

#### layer
> Layer(int id, int prev_node_count, int node_count, InitializerBase* initialization_function, ActivationFuncBase* activation_function, OptimizingFuncBase* optimizer_function, RegularizationFuncBase* reg_func, float probability_dropout)
> ```
> id: Unique Layer ID
> prev_node_count: No. of nodes in the previous layer
> node_count: No. of nodes in the layer being defined
> initialization_function: Initialize Function's pointer
> activation_function: Activation Function's pointer, nullptr -> no activation
> optimizer_function: Optimizer Function's pointer
> reg_func: Regularization Function's pointer
> probability_dropout: probability to drop nodes out, put 0 to stop drop out
> ```

---

#### initializer
> XavierNormal()
> ```
> no parameters
> ```

> XavierUniform()
> ```
> no parameters
> ```

> HeNormal()
> ```
> no parameters
> ``````

> HeUniform()
> ```
> no parameters
> ```

---

#### activations
> Sigmoid()
> ```
> no parameters
> ```

> Tanh()
> ```
> no parameters
> ```

> ReLU()
> ```
> no parameters
> ```

> LeakyReLU()
> ```
> no parameters
> ```

---

#### losses
> BinaryCrossEntropy()
> ```
> no parameters
> ```

> SoftmaxCategoricalCrossEntropy()
> ```
> no parameters (use None for activation function)
> ```

> MeanSquaredLoss()
> ```
> no parameters
> ```

---

#### optimizer
> StochasticGradientDescent(float LR)
> ```
> LR: affects change in weights and biases
> ```

> SGDMomentum(float momentum_coeff, float LR)
> ```
> momentum_coeff: affects how much of the previous momentum is retained
> LR: affects change in weights and biases
> ```

> NesterovAcceleratedGradient(float momentum_coeff, float LR)
> ```
> momentum_coeff: affects how much of the previous momentum is retained
> LR: affects change in weights and biases
> ```

> RMSProp(float decay_rate, float LR)
> ```
> decay_rate: affects how fast the change in weights and biases change
> LR: affects change in weights and biases
> ```

> Adam(float first_moment_decay_rate, float second_moment_decay_rate, float LR)> 
> ```
> first_moment_decay_rate: affects mean
> second_moment_decay_rate: affects variance
> LR: affects change in weights and biases
> ```

---

#### learning_rate_decay
> StepDecay(float initial_lr, float min_lr, int decay_step_size, float decay_factor)
> ```
> initial_lr: initial learning rate to start with
> min_lr: minimum learning rate allowed
> decay_step_size: updates learning rate every `decay_step_size` steps
> decay_factor: affects how drastically learning rate is changed
> ```

> ExponentialDecay(float initial_lr, float min_lr, float decay_constant)
> ```
> initial_lr: initial learning rate to start with
> min_lr: minimum learning rate allowed
> decay_constant: affects how drastically learning rate is changed
> ```

> LinearDecay(float initial_lr, float min_lr, int total_epoch)
> ```
> initial_lr: initial learning rate to start with
> min_lr: minimum learning rate allowed
> total_epoch: no. of epoch
> ```

> CosineAnnealing(float initial_lr, float min_lr, int total_epoch)
> ```
> initial_lr: initial learning rate to start with
> min_lr: lower bound for the lerning rate
> total_epoch: no. of epoch
> ```

#### regularization
> L1Regularization(float lambda_value)
> ```
> Lambda: affects how much weights are affected 
> ```

> L2Regularization(float lambda_value)
> ```
> Lambda: affects how much weights are affected 
> ```

> ElasticNet(float lambda_value)
> ```
> Lambda: affects how much weights are affected 
> ```

## Example program [MNIST DATASET]
```
#include "header.hpp"
#include <fstream>
#include <vector>

// chat gpt ahhhh code cuz i was too lazy to make my own mnist loader 
// ==============================================================================================================================
uint32_t read_uint32(ifstream &ifs) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
        result = (result << 8) | ifs.get();
    return result;
}

void load_mnist_images(const string &filename, vector<vector<float>> &images) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file: " + filename);

    uint32_t magic = read_uint32(file);
    uint32_t num_images = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    images.resize(num_images, vector<float>(rows * cols));

    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel = file.get();
            images[i][j] = pixel / 255.0f;
        }
    }
}

void load_mnist_labels(const string &filename, vector<uint8_t> &labels) {
    ifstream file(filename, ios::binary);
    if (!file) throw runtime_error("Cannot open file: " + filename);

    uint32_t magic = read_uint32(file);
    uint32_t num_labels = read_uint32(file);

    labels.resize(num_labels);

    for (uint32_t i = 0; i < num_labels; ++i) {
        labels[i] = file.get();
    }
}

void one_hot_encode(const vector<uint8_t>& labels, vector<vector<float>>& one_hot_labels, int num_classes = 10) {
    float epsilon = 0.01f;
    one_hot_labels.resize(labels.size(), vector<float>(num_classes, epsilon));
    for (size_t i = 0; i < labels.size(); i++) {
        one_hot_labels[i][labels[i]] = 1.0f - (num_classes-1) * epsilon;
        // cout << to_string(labels[i]) << ": " << Print1DVector(one_hot_labels[i]);
    }

}
// ==============================================================================================================================

int main() {
    // define params
    float initial_lr = 0.001f;
    float min_lr = 1e-5f;
    float Pdropout = 0.02f;
    float lambda_value_reg = 1e-4f;
    int total_epoch = 10;
    int batch_size = 30;
    float momentumcoeff = 0.9f;
    float fmdr = 0.9f;
    float smdr = 0.999f;

    // define model
    Network model(0);
    model.add_Layer(new Layer(0, 784, 512, new HeNormal(), new ReLU(), new Adam(fmdr, smdr, initial_lr), new ElasticNet(lambda_value_reg), Pdropout));
    model.add_Layer(new Layer(1, 512, 256, new HeNormal(), new ReLU(), new Adam(fmdr, smdr, initial_lr), new ElasticNet(lambda_value_reg), Pdropout));
    model.add_Layer(new Layer(2, 256, 128, new HeNormal(), new ReLU(), new Adam(fmdr, smdr, initial_lr), new ElasticNet(lambda_value_reg), Pdropout));
    model.add_Layer(new Layer(3, 128, 10,  new HeNormal(), nullptr,    new Adam(fmdr, smdr, initial_lr), new ElasticNet(lambda_value_reg), 0.0f));
    model.compile_network(new SoftmaxCategoricalCrossEntropy(), new CosineAnnealing(initial_lr, min_lr, total_epoch), batch_size);

    // define data
    vector<vector<float>> train_images;
    vector<uint8_t> train_labels_raw;
    vector<vector<float>> train_labels_one_hot;
    vector<vector<float>> test_images;
    vector<uint8_t> test_labels_raw;
    vector<vector<float>> test_labels_one_hot;
    load_mnist_images("./samples/train-images-idx3-ubyte", train_images);
    load_mnist_labels("./samples/train-labels-idx1-ubyte", train_labels_raw);
    one_hot_encode(train_labels_raw, train_labels_one_hot);
    load_mnist_images("./samples/t10k-images-idx3-ubyte", test_images);
    load_mnist_labels("./samples/t10k-labels-idx1-ubyte", test_labels_raw);
    one_hot_encode(test_labels_raw, test_labels_one_hot);

    // train and evaluate model
    // Model train mai problem hai fuck
    model.train(train_images, train_labels_one_hot, total_epoch);
    float acc = model.Evaluate(test_images, test_labels_one_hot);
    cout << "Accuracy: " << acc << "%" << endl ;
    return 0;
}
```
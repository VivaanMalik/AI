#include "header.hpp"

Network::Network(int id) : 
    id(id),
    Layers({}),
    LossFunction(nullptr),
    LearningRateDecayFunction(nullptr),
    EpochNumber(0),
    RegularizationFunction(nullptr),
    BatchSize(-1) {
    }

void Network::add_Layer(Layer layer) {
    Layers.push_back(layer);
}

void Network::compile_network(LossFuncBase* loss_func, LearningRateDecayFuncBase* lrd_func, RegularizationFuncBase* reg_func, int batch_size) {
    BatchSize = batch_size;
    if (loss_func)
        LossFunction = loss_func;
    if (lrd_func)
        LearningRateDecayFunction = lrd_func;
    if (reg_func)
        RegularizationFunction = reg_func;
    for (int i = 0; i < Layers.size(); i++) {
        Layers[i].initialize(BatchSize);
    }
}

float* Network::forward(float* x) {
    for (int i = 0; i < Layers.size(); i++) {
        x = Layers[i].forward(x);
    }
    return x;
}

void Network::backward(float* grad) {
    for (int i = Layers.size(); i>0; i--) {
        grad = Layers[i].backward(grad);
    }
}

template<typename T>
void setTotalEpochIfPossible(T& obj, int value) {
    if constexpr (requires(T x) { x->setTotalEpoch(value); }) {
        obj->setTotalEpoch(value);
    }
}

__global__ void argmax_and_count_matches(float* prediction, float* target, int* correct_count, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Argmax for prediction
    int pred_idx = 0;
    float pred_max = prediction[row * cols];
    for (int i = 1; i < cols; i++) {
        float val = prediction[row * cols + i];
        if (val > pred_max) {
            pred_max = val;
            pred_idx = i;
        }
    }

    // Argmax for target
    int truth_idx = 0;
    float truth_max = target[row * cols];
    for (int i = 1; i < cols; i++) {
        float val = target[row * cols + i];
        if (val > truth_max) {
            truth_max = val;
            truth_idx = i;
        }
    }

    // Count match
    if (pred_idx == truth_idx) {
        atomicAdd(correct_count, 1);
    }
}

void Network::train(vector<vector<float>> input_data, <vector<vector<float>> output_target_data, int epoch) {
    if (LearningRateDecayFunction)
        setTotalEpochIfPossible(LearningRateDecayFunction, epoch);

    float* input_data_batch;
    float* output_target_data_batch;
    int* num_of_correct_predictions;

    int datasize = input_data.size();
    int inputsize = input_data[0].size();
    int outputsize = output_target_data[0].size();

    cudaMalloc(&input_data_batch, BatchSize * inputsize * sizeof(float));
    cudaMalloc(&input_data_batch, BatchSize * outputsize * sizeof(float));
    cudaMalloc(&num_of_correct_predictions, sizeof(int));

    int CurrentEpochNumber = 0;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    float total_time = 0.0f;
    for (int e = 0; e < epoch; e++) {
        CurrentEpochNumber++;

        // shuffle
        vector<int> indices(datasize);
        for (int i = 0; i < datasize; i++) {
            indices[i] = i;
        }
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);
        vector<vector<float>> tmp_shuffled_input_data;
        vector<vector<float>> tmp_shuffled_output_data;
        for (int i = 0; i < datasize; i++) {
            tmp_shuffled_input_data[i] = move(input_data[indices[i]]);
            tmp_shuffled_output_data[i] = move(output_target_data[indices[i]]);
        }
        input_data = move(tmp_shuffled_input_data);
        output_target_data = move(tmp_shuffled_output_data);

        vector<float> input_batch(BatchSize);
        vector<float> target_batch(BatchSize);
        float* predicted;
        float* grad;
        float* loss;

        cudaMemset(num_of_correct_predictions, 0, sizeof(int));
        for (int b = 0; b < datasize; b+=BatchSize) {
            input_batch.assign(input_data.begin() + b, input_data.begin() + b + BatchSize);
            target_batch.assign(target_batch.begin() + b, target_batch.begin() + b + BatchSize);

            // =============================================================================================
            // INSERT NAG KA CODE HERE MUEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHE
            // =============================================================================================

            cudaMemcpy(input_batch.data(), input_data_batch, BatchSize * inputsize * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(target_batch.data(), output_target_data_batch, BatchSize * outputsize * sizeof(float), cudaMemcpyDeviceToHost);

            predicted = this->forward(input_data_batch);
            loss = LossFunction->forward(predicted, output_target_data_batch, BatchSize, outputsize);

            // =============================================================================================
            // INSERT EVALUATION METRICS KA CODE HERE HEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHE
            int threads = 128;
            int blocks = (BatchSize + threads - 1) / threads;
            argmax_and_count_matches<<<blocks, threads>>>(predicted, output_target_data_batch, num_of_correct_predictions, BatchSize, outputsize);
            // =============================================================================================
            
            grad = LossFunction->backward();
            this->backward(grad);

            if (RegularizationFunction) {
                for (int l = 0; l < Layers.size(); l++) {
                    loss+=RegularizationFunction->UpdateLoss(Layers[l].weights, loss, Layers[l].PrevNodeCount * Layers[l].NodeCount);
                    RegularizationFunction->UpdateGradient(Layers[l].weights, grad, Layers[l].PrevNodeCount * Layers[l].NodeCount);
                }
            }

            // =============================================================================================
            // INSERT SOME MORE NAG KA CODE HERE MUEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHE
            // =============================================================================================

            for (int l = 0; l < Layers.size(); l++) {
                Layers[l].OptimizerFunctionClass->step(Layers[l].weights, Layers[l].biases, Layers[l].dW, Layers[l].dB, Layers[l].PrevNodeCount, Layers[l].NodeCount);
                if (LearningRateDecayFunction) {
                    Layers[l].OptimizerFunctionClass->SetNewLR(LearningRateDecayFunction->decay(e));
                }
            }
        }

        // =============================================================================================
        // INSERT EVALUATION METRICS KA CODE HERE HEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHEHE
        int h_correct;
        cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
        float accuracy = static_cast<float>(h_correct) / datasize;

        float h_loss;
        cudaMemcpy(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);

        float elapsed_time = GetElapsedTime(start);
        total_time+=elapsed_time;
        float h_loss;
        cout << "Epoch: " + to_string(epoch) + " | Loss: " + to_string(round_to_sigfigs(h_loss, 5)) + " | Accuracy: " + to_string(round_to_sigfigs(accuracy, 4))+"% | " + to_string(elapsed_time) + "s\n";
        start = chrono::steady_clock::now();
        // =============================================================================================
    }
    cout << "Training ended in: "+to_string(round_to_sigfigs(total_time, 4))+"s\n";
}

void Network::predict(vector<float> input) {
    
}


// TEST
// ========================================================================================================================================

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
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
    one_hot_labels.resize(labels.size(), vector<float>(num_classes, 0.0f));
    for (size_t i = 0; i < labels.size(); ++i) {
        one_hot_labels[i][labels[i]] = 1.0f;
    }
}
// ==============================================================================================================================

void Program(int id) {
    // define params
    cout<<id<<endl;
    float initial_lr = 0.01f;
    float min_lr = 1e-4f;
    float Pdropout = 0.2f;
    float lambda_value_reg = 0.01f;
    int total_epoch = 10;
    int batch_size = 32;

    // define model
    Network model(id);
    model.add_Layer(new Layer(0, 784, 256, new HeNormal(), new LeakyReLU(), new StochasticGradientDescent(initial_lr), Pdropout));
    model.add_Layer(new Layer(1, 256, 128, new HeNormal(), new LeakyReLU(), new StochasticGradientDescent(initial_lr), Pdropout));
    model.add_Layer(new Layer(2, 128, 64,  new HeNormal(), new LeakyReLU(), new StochasticGradientDescent(initial_lr), Pdropout));
    model.add_Layer(new Layer(3, 64, 10,   new HeNormal(), nullptr,         new StochasticGradientDescent(initial_lr), 0.0f));
    model.compile_network(new SoftmaxCategoricalCrossEntropy(), new CosineAnnealing(initial_lr, min_lr, total_epoch), new ElasticNet(lambda_value_reg), batch_size);

    // define data
    vector<vector<float>> train_images;
    vector<uint8_t> train_labels_raw;
    vector<vector<float>> train_labels_one_hot;
    vector<vector<float>> test_images;
    vector<uint8_t> test_labels_raw;
    vector<vector<float>> test_labels_one_hot;
    load_mnist_images("samples\\train-images-idx3-ubyte", train_images);
    load_mnist_labels("samples\\train-labels-idx1-ubyte", train_labels_raw);
    one_hot_encode(train_labels_raw, train_labels_one_hot);
    load_mnist_images("samples\\t10k-images-idx3-ubyte", test_images);
    load_mnist_labels("samples\\t10k-labels-idx1-ubyte", test_labels_raw);
    one_hot_encode(test_labels_raw, test_labels_one_hot);

    // train and evaluate model
    // Model train mai problem hai
    model.train(train_images, train_labels_one_hot, total_epoch);
    cout << "Accuracy: " << model.Evaluate(test_images, test_labels_one_hot) << endl ;
}
#include "header.hpp"

Network::Network(int id) : 
    id(id),
    Layers({}),
    Initializer(nullptr),
    LossFunction(nullptr),
    LearningRateDecayFunction(nullptr),
    EpochNumber(0),
    WeightDecayFunction(nullptr) {}

void Network::add_Layer(Layer layer) {
    this->Layers.push_back(layer);
}

void Network::log_work(sqlite3* db_pointer, chrono::steady_clock::time_point start) {
    float created_on = GetElapsedTime(start);
    string values = to_string(id) + ", \"Bing Bong" + to_string(id) + "\", " + to_string(created_on);
    write_to_sqlite3_db(db_pointer, values);
    json result = keepchecking(id, db_pointer);

    this->the_thing_to_be_done(result["name"]);

    // float created_on = GetElapsedTime(start);
    // string pre_activation_values_as_str = VectorFLoatToString(pre_activation_values);
    // string values = to_string(id) + ", \"0000 000 " + pre_activation_values_as_str + "\", " + to_string(created_on);
    // write_to_sqlite3_db(db_pointer, values);
    // json result = keepchecking(id, db_pointer);

    // this->output = result["Output"].get<vector<float>>();
    // return this->output;
}

void Network::the_thing_to_be_done(string msg) {
    Sigmoid function;
    vector<float> v = { 0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f,
                        0.1f, 0.2f, 0.3f, 0.4f};
    int batch_size = 4; // vertical
    int feature_size = 4; // horizontal
    int total_size = batch_size * feature_size;
    
    float* result_pointer;

    result_pointer = function.forward(v, batch_size, feature_size);
    string actual_out = Print2DMatrix(unflatten(to_cpu(result_pointer, total_size), batch_size, feature_size));
    cout <<  msg + "\n" + actual_out + "\n";

    result_pointer = function.backward(v, batch_size, feature_size);
    actual_out = Print2DMatrix(unflatten(to_cpu(result_pointer, total_size), batch_size, feature_size));
    cout <<  msg + "\n" + actual_out + "\n";

    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    result_pointer = function.forward(v, batch_size, feature_size);
    result_pointer = function.backward(v, batch_size, feature_size);
    float elapsed_time = GetElapsedTime(start);
    
    cout << "OUTPUT TIME: " + to_string(elapsed_time) + "\n";
}

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
    Sigmoid s;
    vector<vector<float>> v = { {0.1f, 0.2f, 0.3f, 0.4f},
                                {0.1f, 0.2f, 0.3f, 0.4f},
                                {0.1f, 0.2f, 0.3f, 0.4f},
                                {0.1f, 0.2f, 0.3f, 0.4f}};
        
    vector<vector<float>> result =  s.forward(v);
    string actual_out = Print2DMatrix(result);
    cout <<  msg + "\n" + actual_out + "\n";

    result = s.backward(v);
    actual_out = Print2DMatrix(result);
    cout <<  msg + "\n" + actual_out + "\n";
}

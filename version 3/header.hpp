#ifndef HEADER_HPP
#define HEADER_HPP

#include <iostream>
#include <string>
#include <thread>
#include <sqlite3.h>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;

// Initializer
// ===============================================================================
class InitializerBase {
    public:
    virtual json initialize(vector<int> shape) const = 0;
    virtual ~InitializerBase() {}
};

// TODO: add the funcs

// ===============================================================================

// Loss func
// ===============================================================================
class LossFuncBase {
    public:
    virtual float forward(vector<float> output, vector<float> target_output) const = 0;
    virtual vector<float> backward() const = 0;
    virtual ~LossFuncBase() {}
};
// ===============================================================================

// lr decay func
// ===============================================================================
class LearningRateDecayFuncBase {
    public:
    LearningRateDecayFuncBase(float initial_lr);
    virtual float decay(int timestep) const = 0;
    virtual ~LearningRateDecayFuncBase() {}
};
// ===============================================================================

// weight decay func (add loss)
// ===============================================================================
class WeightDecayFuncBase {
    public:
    WeightDecayFuncBase(float lambda);
    virtual float GetAdditionalLoss(vector<vector<float>> weights) const = 0;
    virtual ~WeightDecayFuncBase() {}
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
    WeightDecayFuncBase* WeightDecayFunction;    
    Network(int id);

    void add_Layer(Layer layer);
    void log_work(sqlite3* db_pointer, float created_on);
    void the_thing_to_be_done(std::string msg);
};

sqlite3* setup_sqlite3_db();
void close_sqlite3_db(sqlite3*);
bool write_to_sqlite3_db(sqlite3*, string);
optional<json> read_from_sqlite3_db(sqlite3*, int);
void the_thing_to_be_done(string);
json keepchecking(int, sqlite3*);
json ParseAndComputeData(string);

#endif
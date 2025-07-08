#include "header.hpp"

float GetElapsedTime(chrono::steady_clock::time_point start) {
    float elapsed_time = chrono::duration<float>(chrono::steady_clock::now() - start).count();
    return elapsed_time;
}

string VectorFLoatToString(vector<float> data) {
    string s = "|";
    for (int i = 0; i < data.size(); i++) {
        s+=to_string(data[i]);
        s+=",";
    }
    if (!s.empty()) {
        s.pop_back();
    }
    s+="|";
    return s;
}

string Print2DMatrix(vector<vector<float>> result) {
    string actual_out = "";
    for (int i = 0; i<result.size(); i++) {
        string out = "(";
        for (int j = 0; j<result[i].size(); j++) {
            out+=to_string(result[i][j]);
            out+=", ";
        }
        if (!out.empty()) {
            out.pop_back();
            out.pop_back();
        }
        out+=")";
        actual_out+=out;
        actual_out+="\n";
    }
    return actual_out;
}

string Print1DVector(vector<float> result) {
    string out = "(";
    for (int j = 0; j<result.size(); j++) {
        out+=to_string(result[j]);
        out+=", ";
    }
    if (!out.empty()) {
        out.pop_back();
        out.pop_back();
    }
    out+=")";
    out+="\n";
    return out;
}

vector<float> flatten(vector<vector<float>> input) {
    int thing1 = input.size();
    int thing2 = input[0].size();
    int total_size = thing1 * thing2;
    vector<float> flat_input(total_size);
    for (int i = 0; i < thing1; ++i)
        for (int j = 0; j < thing2; ++j)
            flat_input[i * thing2 + j] = input[i][j];
    return flat_input;
}

vector<vector<float>> unflatten(vector<float> input, int thing1size, int thing2size) {
    vector<vector<float>> result(thing1size, vector<float>(thing2size));
    for (int i = 0; i < thing1size; ++i)
        for (int j = 0; j < thing2size; ++j)
            result[i][j] = input[i * thing2size + j];
    return result;
}
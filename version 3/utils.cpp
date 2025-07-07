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
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

class Network {
public:
    int id;
    Network(int id);
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
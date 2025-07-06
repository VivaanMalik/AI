#include "header.hpp"

Network::Network(int id) : id(id) {}

void Network::log_work(sqlite3* db_pointer, float created_on) {

    string values = to_string(id) + ", \"Bing Bong" + to_string(id) + "\", " + to_string(created_on);
    write_to_sqlite3_db(db_pointer, values);
    json result = keepchecking(id, db_pointer);

    
    
    this->the_thing_to_be_done(result["name"]);
}

void Network::the_thing_to_be_done(string msg) {
    cout <<  msg + "\n\n";
}

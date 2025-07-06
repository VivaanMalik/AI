#include "header.hpp"

int main(){
    int num_of_threads = 8;

    sqlite3* db_pointer = setup_sqlite3_db();

    auto start = chrono::steady_clock::now();

    vector<thread> threads;
    vector<Network> Networks;

    for (int i = 0; i<num_of_threads; i++) {
        Networks.emplace_back(i);

        float elapsed_time = chrono::duration<float>(chrono::steady_clock::now() - start).count();

        threads.emplace_back([&, i, elapsed_time]() {
            Networks[i].log_work(db_pointer, elapsed_time);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    close_sqlite3_db(db_pointer);
    return 0;
 }
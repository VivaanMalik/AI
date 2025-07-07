#include "header.hpp"

// nvcc -o executable .\main.cpp .\SQLiteManager.cpp .\thread.cpp .\Parser.cpp .\Layer.cpp .\activations.cu .\Initializer.cu .\utils.cpp -lsqlite

int main(){
    int num_of_threads = 8;

    sqlite3* db_pointer = setup_sqlite3_db();

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    vector<thread> threads;
    vector<Network> Networks;

    for (int i = 0; i<num_of_threads; i++) {
        Networks.emplace_back(i);

        float elapsed_time = GetElapsedTime(start);

        threads.emplace_back([&, i, elapsed_time]() {
            Networks[i].log_work(db_pointer, start);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    close_sqlite3_db(db_pointer);
    return 0;
}
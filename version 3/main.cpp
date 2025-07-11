#include "header.hpp"
#include <process.h>

int main(int argc, char* argv[]) {
    if (argc > 1) {
        // This is a spawned process
        int id = stoi(argv[1]);
        chrono::steady_clock::time_point start = chrono::steady_clock::now();

        Network net(id);
        net.log_work(start);

    } 
    else {
        // This is the master process
        const int num_processes = 8;
        vector<int> pids;

        for (int i = 0; i < num_processes; ++i) {
            string id_str = to_string(i);

            int result = _spawnl(
                _P_NOWAIT,            // Do not wait
                "build\\executable.exe",     // Path to your compiled exe
                "executable.exe",     // argv[0]
                id_str.c_str(),       // argv[1]
                nullptr               // End
            );

            if (result == -1) {
                cerr << "Failed to launch process " << i << "\n";
            }
            else {
                pids.push_back(result);
            }
        }

        cout << "Spawned all processes.\n";

        for (int pid : pids) {
            _cwait(nullptr, pid, 0);
        }

        cout << "All child processes finished.\n";
    }

    return 0;

}
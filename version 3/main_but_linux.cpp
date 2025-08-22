#include "header.hpp"
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <stdlib.h>

int main(int argc, char **argv) {

    // if (argc>1) {
    //     // cout<< "heyo\n";
    //     Program(atoi(argv[1]));
    //     return 0;
    // }

    // int num_processes = 2;
    // cout << "[Parent " << getpid() << "] spawning " << num_processes << " children...\n";

    // for (int i = 0; i < num_processes; i++) {
    //     pid_t pid = fork();
    //     if (pid == 0) {
    //         execl("./build/executable", "./build/executable",
    //               to_string(i).c_str(), (char*)nullptr);
    //         perror("execl failed");
    //         _exit(1);
    //     }
    // }

    // // Parent waits for all children (not one by one in loop)
    // for (int i = 0; i < num_processes; i++) {
    //     int status;
    //     pid_t cpid = wait(&status);
    //     cout << "[Parent] child " << cpid << " finished\n";
    // }

    // return 0;

    Program(0);
}
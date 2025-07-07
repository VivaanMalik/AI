@REM x64 Native Tools Command Prompt for VS 2022 

@echo off
nvcc -std=c++17 -I./sqlite3 sqlite3/sqlite3.c -o executable .\main.cpp .\SQLiteManager.cpp .\thread.cpp .\Parser.cpp .\Layer.cpp .\activations.cu .\Initializer.cu .\utils.cpp
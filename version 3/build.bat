@REM x64 Native Tools Command Prompt for VS 2022 
@REM cd C:\Users\Vivaan\Documents\Programming\AI\version 3

@echo off
echo Deleting previous executable.exe...
del /f /q build\executable.exe >nul 2>&1
nvcc -std=c++17 -o .\build\executable .\main.cpp .\thread.cpp .\Layer.cpp .\activations.cu .\Initializer.cu .\utils.cpp .\regularization.cu .\lrdecay.cpp
@REM build\executable.exe
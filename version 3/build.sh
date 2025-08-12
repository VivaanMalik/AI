echo "Deleting previous executable.exe..."
rm -f "./build/executable"
nvcc -std=c++17 -o ./build/executable ./TheACTUALprog.cpp ./utils.cpp ./utils.cu ./activations.cu ./optimizer.cu ./loss.cu ./Initializer.cu ./regularization.cu ./main_but_linux.cpp ./network.cu ./Layer.cu ./lrdecay.cpp
./run.sh
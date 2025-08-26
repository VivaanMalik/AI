echo "Deleting previous executable"
rm -f "./build/executable"
nvcc -std=c++17 -o ./build/executable ./main_but_linux.cpp ./utils.cpp ./utils.cu ./activations.cu ./optimizer.cu ./loss.cu ./Initializer.cu ./regularization.cu ./network.cu ./Layer.cu ./lrdecay.cpp
./run.sh
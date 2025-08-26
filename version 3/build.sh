echo "Deleting previous executable"
rm -f "./build/executable"
nvcc -std=c++17 -o ./build/executable ./main_but_linux.cpp ./src/utils.cpp ./src/utils.cu ./src/activations.cu ./src/optimizer.cu ./src/loss.cu ./src/Initializer.cu ./src/regularization.cu ./src/network.cu ./src/Layer.cu ./src/lrdecay.cpp
./run.sh
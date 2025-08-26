#!/bin/bash
set -e

echo "Cleaning old build..."
rm -rf build
mkdir -p build

echo "Building shared library..."
nvcc -Xcompiler -fPIC -shared ./src/utils.cpp ./src/utils.cu ./src/activations.cu ./src/optimizer.cu ./src/loss.cu ./src/Initializer.cu ./src/regularization.cu ./src/network.cu ./src/Layer.cu ./src/lrdecay.cpp -o ./build/libAILib.so
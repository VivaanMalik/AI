echo "Building executable..."
g++ main_but_linux.cpp -Isrc -Lbuild -lAILib -Wl,-rpath=build -o build/executable

echo "Running program..."
./build/executable

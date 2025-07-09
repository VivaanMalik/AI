import NeuralNetwork as nn
import cupy as cp
import time
model = nn.NeuralNetwork()
v = cp.array([[0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4]])

function = nn.LeakyReLU()

start_time = time.time()
result = function.forward(v)
result = function.backward(v)
t = time.time()
print(t-start_time)
start_time = time.time()
result = function.forward(v)
result = function.backward(v)
t = time.time()
print(t-start_time)

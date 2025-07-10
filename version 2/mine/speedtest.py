import NeuralNetwork as nn
import cupy as cp
import time
model = nn.NeuralNetwork()
v = cp.array([[0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.3, 0.4]])

def test_activation_unc():
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

def test_initializing_func():
    function = nn.He()
    start_time = time.time()
    result = function.initialize((4, 8))
    t = time.time()
    print(t-start_time)
    start_time = time.time()
    result = function.initialize((4, 8))
    t = time.time()
    print(t-start_time)

test_initializing_func()

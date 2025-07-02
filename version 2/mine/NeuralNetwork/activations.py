import cupy as cp

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = cp.where(x >= 0, 1 / (1 + cp.exp(-x)), cp.exp(x) / (1 + cp.exp(x)))
        return self.output
    
    def backward(self, grad):
        return grad * self.output*(1-self.output)

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return cp.where(self.mask, x, 0)
    
    def backward(self, grad):
        return grad * cp.where(self.mask, 1, 0)

class LeakyReLU:
    def __init__(self, alpha = 0.01):
        self.mask = None
        self.alpha = alpha

    def forward(self, x):
        self.mask = x > 0
        return cp.where(self.mask, x, self.alpha*x)
    
    def backward(self, grad):
        return grad * cp.where(self.mask, 1, self.alpha)

def FindActivation(name):
    name = name.lower()
    if name == "nonetype":
        return None
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "relu":
        return ReLU()
    elif name == "leakyrelu":
        return LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
import cupy as cp
        
class BinaryCrossEntropy:
    def __init__(self):
        self.pred = None
        self.target = None
        self.eps = 1e-9

    def forward(self, pred, target):
        self.pred = cp.clip(pred, self.eps, 1 - self.eps) # prevent log(0)
        self.target = target
        return -cp.mean(target * cp.log(self.pred) + (1 - target) * cp.log(1 - self.pred))

    def backward(self):
        return (self.pred - self.target) / (self.pred * (1 - self.pred) * self.target.size)

class SoftmaxCategoricalCrossEntropy: # use None for activation function
    def __init__(self):
        self.probs = None
        self.target = None

    def forward(self, x, target):
        x_new = x - cp.max(x, axis=1, keepdims=True) # stable
        exp = cp.exp(x_new)
        self.probs = exp / cp.sum(exp, axis=1, keepdims=True)
        self.target = target
        log_probs = -cp.log(cp.clip(cp.sum(self.probs * target, axis=1) + 1e-9, 1e-9, 1.0))
        return cp.mean(log_probs)

    def backward(self):
        # Gradient of fused softmax + crossentropy
        return (self.probs - self.target) / self.target.shape[0]    
    
class MeanSquaredError:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return cp.mean((pred - target) ** 2)

    def backward(self):
        return 2 * (self.pred - self.target) / self.target.size # derivative
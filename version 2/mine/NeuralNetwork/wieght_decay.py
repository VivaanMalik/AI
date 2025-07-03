import cupy as cp

class L2Regularization:
    def __init__(self, Lambda = 1e-4):
        self.Lambda = Lambda
    
    def GetAdditionalLoss(self, weights):
        return self.Lambda * cp.sum(weights**2)
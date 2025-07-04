import cupy as xp

class L2Regularization:
    def __init__(self, Lambda = 1e-4):
        self.Lambda = Lambda
    
    def GetAdditionalLoss(self, weights):
        return self.Lambda * xp.sum(weights**2)
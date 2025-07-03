import xp
from .optimizer import *

class Layer:
    def __init__(self, ID, PreviousLayerNodeCount, CurrentLayerNodeCount, ActivationFunction = None, OptimizingFunction = StochasticGradientDescent(), DropOutProbability = 0):
        self.id = ID
        self.PreviousNodes = PreviousLayerNodeCount
        self.Nodes = CurrentLayerNodeCount
        self.Shape = (self.PreviousNodes, self.Nodes)
        self.ActivationFunction = ActivationFunction
        self.PreActivationValues = xp.array([0]*self.Nodes)
        self.PostActivationValues = xp.array([0]*self.Nodes) 
        self.Initialized = False
        self.Optimizer = OptimizingFunction
        self.isTraining = False
        self.Pdropout = DropOutProbability # around 0.3 to 0.5
    
    # Initializaion functions
    def InitializeNew(self, func):
        self.weights, self.biases = func.initialize(self.Shape)
        self.dW = xp.zeros_like(self.weights)
        self.dB = xp.zeros_like(self.biases)
        self.Initialized = True

    def InitializeOld(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.dW = xp.zeros_like(self.weights)
        self.dB = xp.zeros_like(self.biases)
        self.Initialized = True

    # Propogation functions
    def forward(self, inputvals):
        self.input = inputvals

        self.PreActivationValues = inputvals @ self.weights + self.biases

        if self.ActivationFunction!=None:
            self.PostActivationValues = self.ActivationFunction.forward(self.PreActivationValues)
        else:
            self.PostActivationValues = self.PreActivationValues

        if self.isTraining and self.Pdropout > 0.0:
            self.dropout_mask = (xp.random.rand(*self.PostActivationValues.shape) > self.Pdropout)

            self.PostActivationValues *= self.dropout_mask
            self.PostActivationValues /= (1.0 - self.Pdropout)

        return self.PostActivationValues

    def backward(self, gradient_output):
        # scale grad
        if self.isTraining and self.Pdropout > 0.0:
            gradient_output *= self.dropout_mask
            gradient_output /= (1.0 - self.Pdropout)

        # gradient_output: ∂L/∂a
        if self.ActivationFunction!=None:
            gradient_output = self.ActivationFunction.backward(gradient_output)
        # gradient_output: ∂a/∂z

        self.dW = self.input.T @ gradient_output    # ∂L/∂W = inputᵀ · ∂L/∂z
        self.dB = xp.sum(gradient_output, axis=0, keepdims=True)    # ∂L/∂b = sum over batch dimension

        # ∂L/∂input = ∂L/∂z · Wᵀ
        return gradient_output @ self.weights.T

    def update_wnb(self):
       self.Optimizer.step(self)

    # Debug functions
    def paramters(self):
        return {'w':self.weights, 'b':self.biases}
    
    def gradients(self):
        return {'dW':self.dW, 'dB':self.dB}
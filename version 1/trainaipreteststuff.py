import numpy as np
import json

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class NeuralNetwork:
    def __init__(self, mode = 0,  noofinput = None, layerconfig = None, activationfunction = "sigmoid"):
        """
        mode:
        0 -> usefromjsononly        
        1 -> usefromjsonandupdate  
        2 -> createnewonly          
        3 -> createnewandupdatejson

        layerconfig like      3 4 5 1
        """       

        self.inputcount = noofinput
        self.activationfunc = activationfunction
        if mode == 0 or mode == 1:
            with open("aidatathatithasntreadyetcuzitdumbdumb.json", "r") as f:
                data = json.load(f)
                self.weights = data['weights']
                self.biases = data['biases']
                self.layerconfig = [len(i) for i in data['biases']]
        elif mode == 2 or mode == 3:
            self.layerconfig = layerconfig
            self.weights = []
            self.biases = []
            for i in range(len(layerconfig)-1):
                w = 0.01 * np.random.randn(layerconfig[i], layerconfig[i+1])
                b = np.random.randn(layerconfig[i+1])
                self.weights.append(w)
                self.biases.append(b)
        self.PreActivations = [np.zeros(b.shape) for b in self.biases]
        self.Outputs = [np.zeros(b.shape) for b in self.biases]
        
    def ActivationFunction(self, x):
        if self.activationfunc == "sigmoid":
            return 1/(1+np.exp(-x))
        
    def ActivationFunctionDerivative(self, x):
        if self.activationfunc == "sigmoid":
            s = self.ActivationFunction(x)
            return s * (1 - s)
    
    def forwardprops(self, inputval):
        self.inputvec = np.array(inputval).flatten()
        x = self.inputvec
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(x, w) + b
            a = self.ActivationFunction(z)
            self.PreActivations[i] = z
            self.Outputs[i] = a
            x = a
        return self.Outputs[-1]
    
    def backprop(self, target, learning_rate=0.01):
        # ---------- Backward pass ----------
        # Ensure target and out`put are both 1D arrays
        y_true = np.array(target).flatten()
        y_pred = self.Outputs[-1].flatten()

        # Output layer error
        delta = y_pred - y_true

        # Initialize gradients
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        # Last layer gradients
        a_prev = self.Outputs[-2] if len(self.Outputs) > 1 else self.inputvec
        grad_w[-1] = a_prev.reshape(-1, 1) @ delta.reshape(1, -1)
        grad_b[-1] = delta

        # Backpropagate through hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            dz = self.ActivationFunctionDerivative(self.PreActivations[l])
            delta = (self.weights[l + 1] @ delta) * dz
            a_prev = self.Outputs[l - 1] if l > 0 else self.inputvec
            grad_w[l] = a_prev.reshape(-1, 1) @ delta.reshape(1, -1)
            grad_b[l] = delta

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_w[i]
            self.biases[i] -= learning_rate * grad_b[i]

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork(mode=2, noofinput=2, layerconfig=[2, 2, 1])

# Train for 10000 epochs
for epoch in range(10000):
    total_loss = 0
    for i in range(len(X)):
        out = nn.forwardprops(X[i])
        loss = binary_cross_entropy(y[i], out)
        total_loss += loss
        nn.backprop(y[i], learning_rate=0.1)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {float(total_loss):.4f}")

for sample in X:
    pred = nn.forwardprops(sample)
    print(f"Input: {sample}, Predicted: {np.round(pred)}")
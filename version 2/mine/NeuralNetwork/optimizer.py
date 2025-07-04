import cupy as xp

class StochasticGradientDescent:
    def __init__(self, learning_rate = 0.01, load_param = None):
        self.learning_rate = learning_rate

        if load_param != None:
            self.learning_rate = load_param[0]

    def step(self, layer):
        layer.weights -= self.learning_rate * layer.dW
        layer.biases -= self.learning_rate * layer.dB
    
    def UpdateLearningRate(self, lr):
        self.learning_rate = lr

    def get_param(self):
        return [self.learning_rate]

class SGDMomentum:
    def __init__(self, momentum_coeff = 0.9, learning_rate = 0.01, load_param = None):
        self.momentum_coeff = momentum_coeff
        self.learning_rate = learning_rate
        self.setup = False
        self.weight_velocities = None
        self.bias_velocities = None

        if load_param != None:
            self.momentum_coeff = load_param[0]
            self.learning_rate = load_param[1]
    
    def step(self, layer):

        self.weight_velocities = self.momentum_coeff * self.weight_velocities - self.learning_rate * layer.dW
        self.bias_velocities = self.momentum_coeff * self.bias_velocities - self.learning_rate * layer.dB

        layer.weights += self.weight_velocities
        layer.biases += self.bias_velocities
        
    def UpdateLearningRate(self, lr):
        self.learning_rate = lr
    
    def get_param(self):
        return [self.momentum_coeff, self.learning_rate]

class NesterovAcceleratedGradient:
    def __init__(self, momentum_coeff = 0.9, learning_rate = 0.01, load_param = None):
        self.momentum_coeff = momentum_coeff
        self.learning_rate = learning_rate
        self.setup = False
        self.weight_velocities = None
        self.bias_velocities = None

        if load_param != None:
            self.momentum_coeff = load_param[0]
            self.learning_rate = load_param[1]
    
    def step(self, layer):
        if not self.setup:
            # setup
            self.weight_velocities = xp.zeros_like(layer.weights)
            self.bias_velocities = xp.zeros_like(layer.biases)
            self.setup = True

        self.weight_velocities = self.momentum_coeff * self.weight_velocities - self.learning_rate * layer.dW
        self.bias_velocities = self.momentum_coeff * self.bias_velocities - self.learning_rate * layer.dB

        layer.weights += self.weight_velocities
        layer.biases += self.bias_velocities
        
    def UpdateLearningRate(self, lr):
        self.learning_rate = lr
    
    def get_param(self):
        return [self.momentum_coeff, self.learning_rate]

class RMSProp:
    def __init__(self, decay_rate = 0.9, learning_rate = 0.001, load_param = None):
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.setup = False
        self.epsilon = 1e-9

        self.running_avg_of_squared_weight_gradients = None
        self.running_avg_of_squared_bias_gradients = None

        if load_param != None:
            self.decay_rate = load_param[0]
            self.learning_rate = load_param[1]

    def step(self, layer):
        if not self.setup:
            # setup
            self.running_avg_of_squared_weight_gradients = xp.zeros_like(layer.weights)
            self.running_avg_of_squared_bias_gradients = xp.zeros_like(layer.biases)
            self.setup = True
        
        self.running_avg_of_squared_weight_gradients = self.decay_rate * self.running_avg_of_squared_weight_gradients + (1 - self.decay_rate) * (layer.dW ** 2)
        self.running_avg_of_squared_bias_gradients = self.decay_rate * self.running_avg_of_squared_bias_gradients + (1 - self.decay_rate) * (layer.dB ** 2)

        layer.weights -= self.learning_rate * layer.dW / (xp.sqrt(self.running_avg_of_squared_weight_gradients + self.epsilon))
        layer.biases -= self.learning_rate * layer.dB / (xp.sqrt(self.running_avg_of_squared_bias_gradients + self.epsilon))
        
    def UpdateLearningRate(self, lr):
        self.learning_rate = lr
    
    def get_param(self):
        return [self.decay_rate, self.learning_rate]

class Adam:
    def __init__(self, first_moment_decay_rate = 0.9, second_moment_decay_rate = 0.999, learning_rate = 0.001, timestep = 0, load_param = None):
        self.first_moment_decay_rate = first_moment_decay_rate
        self.second_moment_decay_rate = second_moment_decay_rate
        self.learning_rate = learning_rate
        self.setup = False
        self.epsilon = 1e-9
        self.timestep = timestep

        self.first_moment_weights = None
        self.second_moment_weights = None
        self.first_moment_biases = None
        self.second_moment_biases = None
    
    def step(self, layer):
        if not self.setup:
            # setup
            self.first_moment_weights = xp.zeros_like(layer.weights)
            self.second_moment_weights = xp.zeros_like(layer.weights)
            self.first_moment_biases = xp.zeros_like(layer.biases)
            self.second_moment_biases = xp.zeros_like(layer.biases)
            self.setup = True
        
        self.timestep+=1
        
        # mean
        self.first_moment_weights = self.first_moment_decay_rate * self.first_moment_weights + (1-self.first_moment_decay_rate) * layer.dW
        self.first_moment_biases = self.first_moment_decay_rate * self.first_moment_biases + (1-self.first_moment_decay_rate) * layer.dB
        
        # variance
        self.second_moment_weights = self.second_moment_decay_rate * self.second_moment_weights + (1 - self.second_moment_decay_rate) * (layer.dW ** 2)
        self.second_moment_biases = self.second_moment_decay_rate * self.second_moment_biases + (1 - self.second_moment_decay_rate) * (layer.dB ** 2)        

        bias_corrected_mean_weights = self.first_moment_weights/(1-(self.first_moment_decay_rate ** self.timestep))
        bias_corrected_mean_biases = self.first_moment_biases/(1-(self.first_moment_decay_rate ** self.timestep))

        bias_corrected_variance_weights = self.second_moment_weights/(1-(self.second_moment_decay_rate ** self.timestep))
        bias_corrected_variance_biases = self.second_moment_biases/(1-(self.second_moment_decay_rate ** self.timestep))

        layer.weights -= self.learning_rate * bias_corrected_mean_weights/(xp.sqrt(bias_corrected_variance_weights) + self.epsilon)
        layer.biases -= self.learning_rate * bias_corrected_mean_biases/(xp.sqrt(bias_corrected_variance_biases) + self.epsilon)
    
    def UpdateLearningRate(self, lr):
        self.learning_rate = lr
    
    def get_param(self):
        return [self.first_moment_decay_rate, self.second_moment_decay_rate, self.learning_rate, self.timestep]

def FindOptimizer(name):
    name = name.lower()
    if name == "stochasticgradientdescent":
        return StochasticGradientDescent
    elif name == "sgdmomentum":
        return SGDMomentum
    elif name == "nesterovacceleratedgradient":
        return NesterovAcceleratedGradient
    elif name == "rmsprop":
        return RMSProp
    elif name == "adam":
        return Adam
    else:
        raise ValueError(f"Unknown Optimizing function: {name}")
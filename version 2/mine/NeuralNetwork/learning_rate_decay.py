import math

class StepDecay:
    def __init__(self, initial_lr, decay_step_size, decay_factor = 0.5):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_step_size = decay_step_size
    
    def decay(self, timestep):
        return self.initial_lr * (self.decay_factor ** math.floor(timestep/self.decay_step_size))

class ExponentialDecay:
    def __init__(self, initial_lr, decay_constant = 0.01):
        self.initial_lr = initial_lr
        self.decay_constant = decay_constant

    def decay(self, timestep):
        return self.initial_lr * math.exp(-self.decay_constant*timestep)

class LinearDecay:
    def __init__(self, initial_lr, total_epoch = None):
        self.initial_lr = initial_lr
        self.total_epoch = total_epoch
    
    def SetTotalEpoch(self, T):
        self.total_epoch = T

    def decay(self, timestep):
        return self.initial_lr * (1 - (timestep/self.total_epoch))

class CosineAnnealing:
    def __init__(self, initial_lr, min_lr, total_epoch = None):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epoch = total_epoch
    
    def SetTotalEpoch(self, T):
        self.total_epoch = T
    
    def decay(self, timestep):
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1+math.cos(math.pi * timestep / self.total_epoch))
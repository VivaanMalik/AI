import cupy as cp

class Xavier:
    def initialize(self, shape):
        layerin, layerout = shape
        w = cp.random.randn(layerin, layerout) * cp.sqrt(2.0 / (layerin + layerout))
        b = cp.zeros((1, shape[1]))
        return w, b

class He:
    def initialize(self, shape):
        layerin, layerout = shape
        w = cp.random.randn(layerin, layerout) * cp.sqrt(1.0 / (layerin))
        b = cp.zeros((1, shape[1]))
        return w, b

class XavierUniform:
    def initialize(self, shape):
        layerin, layerout = shape
        limit = cp.sqrt(6.0 / (layerin + layerout))
        w =  cp.random.uniform(-limit, limit, shape)
        b = cp.zeros((1, shape[1]))
        return w, b
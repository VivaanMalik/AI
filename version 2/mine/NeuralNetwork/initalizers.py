import xp

class Xavier:
    def initialize(self, shape):
        layerin, layerout = shape
        w = xp.random.randn(layerin, layerout) * xp.sqrt(2.0 / (layerin + layerout))
        b = xp.zeros((1, shape[1]))
        return w, b

class He:
    def initialize(self, shape):
        layerin, layerout = shape
        w = xp.random.randn(layerin, layerout) * xp.sqrt(1.0 / (layerin))
        b = xp.zeros((1, shape[1]))
        return w, b

class XavierUniform:
    def initialize(self, shape):
        layerin, layerout = shape
        limit = xp.sqrt(6.0 / (layerin + layerout))
        w =  xp.random.uniform(-limit, limit, shape)
        b = xp.zeros((1, shape[1]))
        return w, b
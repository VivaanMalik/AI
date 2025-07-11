import os
import json
import random
import time
import cupy as xp
from .layer import Layer
from .activations import FindActivation
from .optimizer import NesterovAcceleratedGradient, FindOptimizer

class NeuralNetwork():
    def __init__(self):
        self.Layers:list[Layer] = []
        self.Initializer = None
        self.LossFunction = None
        self.LearningRateDecayFunc = None
        self.EpochNumber = 0
        self.TotalEpochNumber = 0
        self.WeightDecayFunc = None
        self._load_data_from_json = False

    def add(self, layer:Layer): 
        self.Layers.append(layer)
    
    def load_data_from_JSON(self, filepath):
        self._load_data_from_json = True

        with open(filepath, "r") as f:
            data = json.load(f)
            self.EpochNumber = data["Epoch"]
            self.TotalEpochNumber = self.EpochNumber

            for i in data["LayerData"]:
                layer = Layer(i["id"], i["PreviousNodeCount"], i["NodeCount"], FindActivation(i["ActivationFunction"]), FindOptimizer(i["OptimizerFunction"])(load_param = i["OptimizerFunctionParameters"]))
                layer.InitializeOld(xp.array(i["weights"]), xp.array(i["biases"]))
                self.Layers.append(layer)
    
    def load_data_to_JSON(self, filepath):
        data = {"Epoch": self.EpochNumber, "LayerData":[]}
        d =[]
        for i in self.Layers:
            d.append({"id":i.id, "PreviousNodeCount": i.PreviousNodes, "NodeCount":i.Nodes, "ActivationFunction":i.ActivationFunction.__class__.__name__, "OptimizerFunction":i.Optimizer.__class__.__name__, "OptimizerFunctionParameters":i.Optimizer.get_param(), "weights":i.weights.tolist(), "biases":i.biases.tolist()})
        data["LayerData"] = d
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def compile_network(self, loss, initializer = None, lrdecayfunc = None, WeightDecayFunc = None):
        self.LossFunction = loss
        self.LearningRateDecayFunc = lrdecayfunc
        self.WeightDecayFunc = WeightDecayFunc
        if not self._load_data_from_json:
            self.Initializer = initializer
            for i in self.Layers:
                if not i.Initialized:
                    i.InitializeNew(self.Initializer)

    def forward(self, x):
        for l in self.Layers:
            x = l.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.Layers):
            grad = layer.backward(grad)

    def train(self, input_values, target_values, epochs, batch_size):
        n = len(input_values)
        self.TotalEpochNumber += epochs

        for l in self.Layers:
            l.isTraining = True
        
        start_time = time.time()
        total_time = 0
        for _current_epoch in range(epochs):
            self.EpochNumber += 1

            # shuffle training dataset
            shuffled = list(zip(input_values, target_values))
            random.shuffle(shuffled)
            input_values, target_values = zip(*shuffled)
            input_values, target_values = list(input_values), list(target_values)
            input_values = xp.array(input_values)
            target_values = xp.array(target_values)

            pred_acc = xp.array([])
            truth_acc = xp.array([])

            for i in range(0, n, batch_size):
                input_batch = input_values[i:i+batch_size]
                target_batch = target_values[i:i+batch_size]

                # ONLY FOR NAG =============================================================
                for layer in self.Layers:
                    if isinstance(layer.Optimizer, NesterovAcceleratedGradient):
                        if not layer.Optimizer.setup:
                            layer.Optimizer.weight_velocities = xp.zeros_like(layer.weights)
                            layer.Optimizer.bias_velocities = xp.zeros_like(layer.biases)
                            layer.Optimizer.setup = True
                        layer.weights += layer.Optimizer.momentum_coeff * layer.Optimizer.weight_velocities
                        layer.biases  += layer.Optimizer.momentum_coeff * layer.Optimizer.bias_velocities
                # ONLY FOR NAG =============================================================

                prediction = self.forward(input_batch)
                loss = self.LossFunction.forward(prediction, target_batch)

                pred=xp.argmax(prediction, axis=1)
                truth = xp.argmax(target_batch, axis = 1)
                pred_acc = xp.concatenate((pred_acc, pred))
                truth_acc = xp.concatenate((truth_acc, truth))

                grad = self.LossFunction.backward()
                self.backward(grad)

                if self.WeightDecayFunc!=None:
                    for layer in self.Layers:
                        loss+=self.WeightDecayFunc.GetAdditionalLoss(layer.weights)

                # ONLY FOR NAG =============================================================
                for layer in self.Layers:
                    if isinstance(layer.Optimizer, NesterovAcceleratedGradient):
                        layer.weights -= layer.Optimizer.momentum_coeff * layer.Optimizer.weight_velocities
                        layer.biases  -= layer.Optimizer.momentum_coeff * layer.Optimizer.bias_velocities
                # ONLY FOR NAG =============================================================
            
                for layer in self.Layers:
                    layer.update_wnb()
                    if self.LearningRateDecayFunc!=None:
                        if hasattr(self.LearningRateDecayFunc, "SetTotalEpoch"):
                            self.LearningRateDecayFunc.SetTotalEpoch(self.TotalEpochNumber)
                        layer.Optimizer.UpdateLearningRate(self.LearningRateDecayFunc.decay(self.EpochNumber))

            accuracy =  float(xp.mean(pred_acc == truth_acc))

            t = time.time()
            timediff = round(t - start_time, 2)
            total_time+=timediff
            print("Epoch: "+str(self.EpochNumber)+" | Loss: "+str(round(float(loss), 5))+ " | Accuracy: "+str(round(accuracy*100, 2))+"% ("+str(timediff)+"s)")
            start_time = t
        
        print("Training ended in: "+str(round(total_time, 2))+"s")
        
        for l in self.Layers:
            l.isTraining = False

    def predict(self, input_vals):
        return xp.argmax(self.forward(input_vals), axis=1)

    def evaluate(self, testinput, testtargetoutput, Metrics:list[str] = ["Accuracy", "ConfusionMatrix"]):
        testinput = xp.array(testinput)
        testtargetoutput = xp.array(testtargetoutput)
        pred=self.predict(testinput)
        truth = xp.argmax(testtargetoutput, axis = 1)

        accuracy = None
        ConfusionMatrix = None

        if "Accuracy" in Metrics:
            accuracy =  float(xp.mean(pred == truth))

        if "ConfusionMatrix" in Metrics:
            pred = pred.tolist()
            truth = truth.tolist()
            n = self.Layers[-1].Nodes
            ConfusionMatrix = xp.zeros((n, n), dtype=int).tolist() 
            for i in range(len(pred)):
                ConfusionMatrix[truth[i]][pred[i]]+=1

        return {"Accuracy": accuracy, "ConfusionMatrix": ConfusionMatrix}
    
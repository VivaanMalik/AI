import cupy as np
import json
import os
import time

class NewralNetwerk:
    def __init__(self, UseFromJSON:bool = False, LayerConfig:list = None, OutputActivationFunction:str = "Sigmoid", HiddenActivationFunction:str = "ReLU", AutoUpdateJSON:bool = False, SaveLocation:bytes = None, LossFunction:str = "BinaryCrossEntropy"):
        """
        UseFromJSON:\n
        True -> usefromjsononly\n      
        False -> createnewonly\n       
        \n
        LayerConfig (includes input layer)\n      
        3 4 5 1 (3 input, 1 output)\n
        \n
        ActivationFunctions:\n
        "Sigmoid", "ReLU", "Softmax", "tanh", "LeakyReLU"\n
        \n
        AutoUpdateTable:\n
        auto updates json after training\n
        \n
        SaveLocation:\n
        json file location\n
        \n
        LossFunction:\n
        "BinaryCrossEntropy" - Binary Cross Entropy\n
        "CategoricalCrossEntropy" - Categorical Cross Entropy\n
        "SquaredDifference" - 0.5 * (difference)^2\n
        "Custom"\n
        """       
        self.Lossfunc = LossFunction
        self.SaveLocation = SaveLocation
        self.updatetable = AutoUpdateJSON
        if UseFromJSON :
            with open(self.SaveLocation, "r") as f:
                data = json.load(f)
                if "Epoch" in data:
                    self.EpochNumber = data["Epoch"]
                else:
                    self.EpochNumber = 0
                self.weights = [np.array(i) for i in data['weights']]
                self.biases = [np.array(i) for i in data['biases']]
                # print(self.weights[0].shape)
                self.LayerConfig = [len(i) for i in data['biases']]
                self.LayerConfig.insert(0, self.weights[0].shape[0])
        else:
            self.EpochNumber = 0
            self.LayerConfig = LayerConfig
            self.weights = []
            self.biases = []
            for i in range(len(LayerConfig)-1):
                w = np.random.randn(LayerConfig[i], LayerConfig[i+1]) * np.sqrt(2.0 / LayerConfig[i])
                b = np.random.randn(LayerConfig[i+1])
                self.weights.append(w)
                self.biases.append(b)
        self.activationfunc = [HiddenActivationFunction]*(len(self.biases)-1)
        self.activationfunc.append(OutputActivationFunction)
        self.PreActivations = [np.zeros(b.shape) for b in self.biases]
        self.Outputs = [np.zeros(b.shape) for b in self.biases]

        # definitions
        self.sigmoid = "Sigmoid"
        self.relu = "ReLU"
        self.leakyrelu = "LeakyReLU"
        self.leakyreluconst = 0.01
        self.softmax = "Softmax"
        self.tanh = "tanh"
        self.binarycf = "BinaryCrossEntropy"
        self.categoricalcf = "CategoricalCrossEntropy"
        self.squaredlf = "SquaredDifference"
        self.customlf = "Custom"
    
    def SetSaveLocation(self, loc:bytes):
        """
        Set the save location (JSON file)
        """
        self.SaveLocation = loc
    
    def SetAutoUpdateTable(self, value:bool):
        """
        Sets the Boolean 'AutoUpdateValue'
        """
        self.updatetable = value
    
    def SetLossFunction(self, f):
        """
        Sets the string 'Lossfunc' to one of the defaults or 'Custom'
        """
        self.Lossfunc = f
    
    def DefineCustomLossFunction(self, func):
        """
        2 arguments - predicted value, target value
        """
        self.CustomLossFunction = func
    
    def SetOutputLayerActivationFunction(self, f):
        """
        Sets Output Layer Activation Function
        """
        self.activationfunc[-1] = f
    
    def SetHiddenLayerActivationFunction(self, f):
        """
        Sets Hidden Layer Activation Function
        """
        for i in range(len(self.activationfunc)-1):
            self.activationfunc[i] = f

    def ActivationFunction(self, x, af):
        if af == self.sigmoid:
            return 1/(1+np.exp(-x))
        elif af == self.relu:
            return np.maximum(0, x)
        elif af == self.softmax:
            x = np.clip(x, -500, 500)
            x_stable = x - np.max(x)
            e_x = np.exp(x_stable)
            softmax = e_x / e_x.sum()
            if np.isnan(softmax).any() or np.isinf(softmax).any():
                print("Softmax instability detected")
                print("Input:", x)
                print("Stable Input:", x_stable)
                raise ValueError("Softmax returned NaN or Inf")
            return softmax
        elif af == self.tanh:
            return np.tanh(x)
        elif af == self.leakyrelu:
            return np.maximum(x*self.leakyreluconst, x)
        else:
            return x
        
    def ActivationFunctionDerivative(self, x, af):
        if af == self.sigmoid or af == self.softmax:
            s = self.ActivationFunction(x, af)
            return s * (1 - s)
        elif af == self.relu:
            return np.where(x>0, 1, 0)
        elif af == self.tanh:
            s = self.ActivationFunction(x, af)
            return (1-s**2)
        elif af == self.leakyrelu:
            return np.where(x>0, 1, self.leakyreluconst)
        else:
            return 1
    
    def Calculate(self, inputval):
        """
        ALternative method for forward propogation due to the creators stupid spelling
        """
        return self.forwardpropoGAYte(inputval)

    def forwardpropoGAYte(self, inputval):
        self.Inputval = np.array(inputval)
        x = self.Inputval
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(x, w) + b
            a = self.ActivationFunction(z, self.activationfunc[i])

            # 🚨 Check for NaN/Inf
            if np.isnan(a).any() or np.isinf(a).any():
                print(f"[FORWARD] NaN or Inf in layer {i} output")
                print(f"Input x: {x}")
                print(f"Z: {z}")
                print(f"A: {a}")
                raise ValueError("Forward pass exploded")
    
            self.PreActivations[i] = z
            self.Outputs[i] = a
            x = a
        return self.Outputs[-1]
    
    def backwordpropoGAYte(self, target, learningrate):
        riyalsolutions = np.array(target)
        aikesolutions = self.Outputs[-1]
        
        errorduetoaikebuttonjaiseaakhe = aikesolutions - riyalsolutions

        if np.isnan(errorduetoaikebuttonjaiseaakhe).any() or np.isinf(errorduetoaikebuttonjaiseaakhe).any():
            print("[BACKPROP] Output layer error has NaN or Inf")
            print("Predicted:", aikesolutions)
            print("Target:", riyalsolutions)
            raise ValueError("Backpropagation exploded")

        GAYdientW = [None] * len(self.weights)
        GAYdientB = [None] * len(self.biases)

        PrevOutput = None
        if len(self.Outputs) > 1:
            PrevOutput = self.Outputs[-2]
        else:
            PrevOutput = self.Inputval
        GAYdientW[-1] = np.dot(PrevOutput.reshape(-1, 1), errorduetoaikebuttonjaiseaakhe.reshape(1, -1))
        GAYdientB[-1] = errorduetoaikebuttonjaiseaakhe

        for l in range(len(self.Outputs)-2, -1, -1):
            derivative = self.ActivationFunctionDerivative(self.PreActivations[l], self.activationfunc[l]) # derivative of post activation value
            errorduetoaikebuttonjaiseaakhe = np.dot(self.weights[l+1], errorduetoaikebuttonjaiseaakhe) * derivative
            if l > 0:
                PrevOutput = self.Outputs[l-1]
            else:
                PrevOutput = self.Inputval
            GAYdientW[l] = np.dot(PrevOutput.reshape(-1, 1), errorduetoaikebuttonjaiseaakhe.reshape(1, -1))
            GAYdientB[l] = errorduetoaikebuttonjaiseaakhe

        for i in range(len(self.Outputs)):
            max_grad = 100.0
            GAYdientW[i] = np.clip(GAYdientW[i], -max_grad, max_grad)
            GAYdientB[i] = np.clip(GAYdientB[i], -max_grad, max_grad)
        
            self.weights[i] -= learningrate * GAYdientW[i]
            self.biases[i] -= learningrate * GAYdientB[i]

    def LossFunctionCalc(self, aiwalabs, riyalval):
        aiwalabs = np.array(aiwalabs)
        riyalval = np.array(riyalval)

        if self.Lossfunc == self.binarycf:
            return self.binary_cross_entropy(aiwalabs, riyalval)
        elif self.Lossfunc == self.categoricalcf:
            return self.categorical_cross_entropy(aiwalabs, riyalval)
        elif self.Lossfunc == self.squaredlf:
            return 0.5 * (aiwalabs - riyalval)**2
        elif self.Lossfunc == self.customlf:
            return float(self.CustomLossFunction(aiwalabs, riyalval))
    
    def binary_cross_entropy(self, aiwalabs, riyalval):
        aiwalabs = np.clip(aiwalabs, 1e-9, 1 - 1e-9)
        return float(-np.sum(riyalval * np.log(aiwalabs) + (1 - riyalval) * np.log(1 - aiwalabs)))

    def categorical_cross_entropy(self, aiwalabs, riyalval):
        aiwalabs = np.clip(aiwalabs, 1e-9, 1 - 1e-9)
        x = float(-np.sum(riyalval * np.log(aiwalabs)))
        return x
            
    def train(self, numofepoch, epochinterval, inputvals, outputvals, lr, inputinterval=None):
        """
        trains the model for \n
        [numofepoch] epochs \n
        with [inputvals] as the input values\n
        and [outputvals] as the target values \n
        with learning rate of [lr]\n
        Also sends updates every [epochinterval] epochs
        """
        self.EpochNumber+=numofepoch
        start = time.process_time()
        for epoch in range(numofepoch):
            total_loss = 0
            for i in range(len(inputvals)):

                if inputinterval!=None:
                    if (i+1)%inputinterval==0:
                        print("input number: "+str(i+1)+" ("+str(time.process_time()-start)+"s)")
                        start = time.process_time()

                out = self.forwardpropoGAYte(inputvals[i])
                loss = self.LossFunctionCalc(out, outputvals[i])
                total_loss += loss

                prev_weights = [w.copy() for w in self.weights]

                self.backwordpropoGAYte(outputvals[i], lr)

                for i in range(len(self.weights)):
                    if np.allclose(prev_weights[i], self.weights[i]):
                        print(f"Layer {i} weights unchanged!")
            
            if (epoch+1) % epochinterval == 0:
                print(f"Epoch {epoch+1} | Loss: {total_loss}")
        
        if self.updatetable:
            self.updatedata()
    
    def TestClassificationModel(self, inputvals, targetoutputval):
        """
        returns \n
        Number of items in datasets: int
        Accuracy:   0 -> 1\n
        Categorical Precision:  [0 -> 1, 0 -> 1, ...]\n
        Categorical Recall: [0 -> 1, 0 -> 1, ...]\n
        Categorical F1 Score: [0 -> 1, 0 -> 1, ...]\n
        Confusion Matrix: [[...], [...], ...]\n
        Average Loss 0 -> inf

        Note:\n
        Number of items in datasets (n)
        Accuracy is Correct / Total\n
        Precision is True Positive / (True Positive + False Positive)\n
        Recall is True Positive / (True Positive + False Negative)\n
        F1 is Harmonic mean of precision and recall\n
        Confusion Matrix is nxn matrix of Categorical Classifications(True Labels on vertical axis
        and Predicted Labels on horizontal axis)\n
        Average Loss represents average loss as described from the set function
        """
        correctoutcomes = 0
        totaloutcomes = len(inputvals)
        aioutput = []
        refinedaioutput = []
        truepositives = [0]*len(targetoutputval[0])
        truenegatives = [0]*len(targetoutputval[0])
        falsepositives = [0]*len(targetoutputval[0])
        falsenegatives = [0]*len(targetoutputval[0])
        ConfusionMatrix = []
        for i in range(len(targetoutputval[0])):
            d = []
            for j in range(len(targetoutputval[0])):
                d.append(0)
            ConfusionMatrix.append(d)
        total_loss = 0
        for i in range(totaloutcomes):
            pred = self.Calculate(inputvals[i]).tolist()
            loss = self.LossFunctionCalc(pred, targetoutputval[i])
            total_loss += loss
            aioutput.append(pred)

            l = [0 for _ in pred]
            maxvalueindex = 0
            for i in range(len(l)):
                if pred[i]>pred[maxvalueindex]:
                    maxvalueindex = i
            l[maxvalueindex] = 1
            refinedaioutput.append(l)

        for i in range(totaloutcomes):
            # Accuracy
            if refinedaioutput[i] == targetoutputval[i]:
                correctoutcomes+=1
            ConfusionMatrix[targetoutputval[i].index(1)][refinedaioutput[i].index(1)]+=1
            
            for j in range(len(targetoutputval[i])):
                if refinedaioutput[i][j] == 1:
                    if targetoutputval[i][j] == 0:
                        falsepositives[j] += 1
                    elif targetoutputval[i][j] == 1:
                        truepositives[j] += 1
                elif refinedaioutput[i][j] == 0:
                    if targetoutputval[i][j] == 0:
                        truenegatives[j] += 1
                    elif targetoutputval[i][j] == 1:
                        falsenegatives[j] += 1

        a = correctoutcomes/totaloutcomes
        cp = [None if (i+j)==0 else i/(i+j) for (i, j) in zip(truepositives, falsepositives)]
        cr = [None if (i+j)==0 else i/(i+j) for (i, j) in zip(truepositives, falsenegatives)]
        cf1 = [None if i==None or j == None or (i+j)==0 else (2*i*j)/(i+j) for (i, j) in zip(cp, cr)]
        return totaloutcomes, a, cp, cr, cf1, ConfusionMatrix, total_loss/len(inputvals)
    
    def updatedata(self):    
        """
        Updates data in the JSON file
        """
        filename = self.SaveLocation   
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as f:
            json.dump({"weights": [i.tolist() for i in self.weights], "biases":[i.tolist() for i in self.biases], "Epoch": self.EpochNumber}, f, indent=4)

def prettyfylistoffloat(attr):
    return ", ".join([str(round(x*100, 2))+"%" if x!=None else "None" for x in attr ])

def prettyfylistofmatrix(attr, labelarray = None):
    if labelarray == None:
        matrix = [[""]]
        for i in range(len(attr)):
            matrix[0].append("Label "+str(i+1))
        for i in range(len(attr)):
            matrix.append(["Label "+str(i+1)]+attr[i])
    else: 
        matrix = [[""]+labelarray]
        for i in range(len(attr)):
            matrix.append([labelarray[i]]+attr[i])

    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return '\n'.join(table)+"\n(Real Label v/s Predicted Label)"
from theACTUALaibase import NewralNetwerk, prettyfylistoffloat, prettyfylistofmatrix
import cupy as np
import json
import random

np.set_printoptions(suppress=True)

def convert_to_int(x):
    x.reverse()
    num = 0
    for j in range(len(x)):
        if x[j] ==1:
            num+=2**j
    return num

def binary_cross_entropy(aiwalabs, riyalval):
    aiwalabs = np.clip(aiwalabs, 1e-9, 1 - 1e-9)
    return -np.sum(riyalval * np.log(aiwalabs) + (1 - riyalval) * np.log(1 - aiwalabs))

filepath = "AITest1\\dataset.json"
data = []
with open(filepath, "r") as f:
    data = json.load(f)

ival = data["4 digit"] + data["5 digit"] + data["6 digit"] + data["7 digit"]
ival = ["{0:b}".format(i) for i in ival]
ival = ["0"*(24 - len(i))+i for i in ival]
ival = [list(i) for i in ival]
ival = [[int(i) for i in j] for j in ival]
oval = [[1, 0, 0, 0]]*1000 + [[0, 1, 0, 0]]*1000 + [[0, 0, 1, 0]]*1000 + [[0, 0, 0, 1]]*1000

shuffled = list(zip(ival, oval))
random.shuffle(shuffled)
ival, oval = zip(*shuffled)

nn = NewralNetwerk(UseFromJSON = False, LayerConfig = [24, 15, 4], SaveLocation = "AITest1\\data.json")
nn.SetAutoUpdateTable(True)
nn.SetLossFunction(nn.categoricalcf)
nn.SetOutputLayerActivationFunction(nn.softmax)
nn.SetHiddenLayerActivationFunction(nn.relu)
nn.train(1, 10, ival, oval, 0.05) # VERY VERY IMP

filepath2 = "AITest1\\dataset2.json"
data = []
with open(filepath2, "r") as f:
    data = json.load(f)
testdata = data["4 digit"] + data["5 digit"] + data["6 digit"] + data["7 digit"]
testdata = ["{0:b}".format(i) for i in testdata]
testdata = ["0"*(24 - len(i))+i for i in testdata]
testdata = [list(i) for i in testdata]
testdata = [[int(i) for i in j] for j in testdata]
testdatatargetoutput = [[1, 0, 0, 0]]*1000 + [[0, 1, 0, 0]]*1000 + [[0, 0, 1, 0]]*1000 + [[0, 0, 0, 1]]*1000

n, accuracy, precision, recall, f1score, confusionmatrix, loss = nn.TestClassificationModel(testdata, testdatatargetoutput)
print("Total Test: " + str(n))
print(str(nn.EpochNumber)+" epochs")
print("Accuracy: " + str(round(accuracy*100, 2))+"%")
print("Categorical Precision: "+prettyfylistoffloat(precision))
print("Categorical Recall: "+prettyfylistoffloat(recall))
print("Categorical F1 Score: "+prettyfylistoffloat(f1score))
print("Confusion Matrix:\n"+prettyfylistofmatrix(confusionmatrix, ["1 digit", "2 digit", "3 digit", "4 digit"]))
print("Average Loss: " + str(round(loss, 5)))
print("Total Loss: " + str(round(loss*n, 5)))

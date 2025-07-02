from theACTUALaibase import NewralNetwerk, prettyfylistoffloat, prettyfylistofmatrix
import cupy as np
from mnist import MNIST
import random

np.set_printoptions(suppress=True)
msgs = ["loading data...", "loaded data!", 
        "coverting labels...", "converted labels!",
        "loading and parsing testing data", "loaded testing data!",
        "Testing and Evaluating...", "Tested and Evaluated!"]

def printfirst(indx):
    print(msgs[indx], end = "")

def eraseandprint(indx):
    print("\b"*len(msgs[indx])+msgs[indx+1]+" "*(len(msgs[indx])-len(msgs[indx+1])))

mnist_data = MNIST("AITest2\\samples")
printfirst(0)
images, labels = mnist_data.load_training()

images = [[j/255.0 for j in i] for i in images]
images = np.array(images)
print("Input mean:", np.mean(images))
print("Input std:", np.std(images))
images = (images - np.mean(images)) / np.std(images)
images = images.tolist()

eraseandprint(0)
printfirst(2)
newlabels = []
for i in labels:
    l = [0]*10
    l[i] = 1
    newlabels.append(l)
eraseandprint(2)

nn = NewralNetwerk(UseFromJSON = False, LayerConfig = [784, 100, 10], SaveLocation = "AITest2\\data.json")
nn.SetAutoUpdateTable(True)
nn.SetLossFunction(nn.categoricalcf)
nn.SetOutputLayerActivationFunction(nn.softmax)
nn.SetHiddenLayerActivationFunction(nn.leakyrelu)
print("training...")
nn.train(9, 1, images[:10000], labels[:10000], lr = 0.01, inputinterval=15000) # VERY VERY IMP
print("training done!")
print("last digit is "+str(labels[-1]))

printfirst(4)

testdata, targetoutput = mnist_data.load_training()
testdata = [[j/255.0 for j in i] for i in testdata]
testdata = np.array(testdata)
testdata = (testdata - np.mean(testdata)) / np.std(testdata)
testdata = testdata.tolist()

testdatatargetoutput = []
for i in targetoutput:
    l = [0]*10
    l[i] = 1
    testdatatargetoutput.append(l)
eraseandprint(4)

printfirst(6)
n, accuracy, precision, recall, f1score, confusionmatrix, loss = nn.TestClassificationModel(testdata, testdatatargetoutput)
eraseandprint(6)
print("Total Test: " + str(n))
print("Accuracy: " + str(round(accuracy*100, 2))+"%")
print("Categorical Precision: "+prettyfylistoffloat(precision))
print("Categorical Recall: "+prettyfylistoffloat(recall))
print("Categorical F1 Score: "+prettyfylistoffloat(f1score))
print("Confusion Matrix:\n"+prettyfylistofmatrix(confusionmatrix, [str(i) for i in range(10)]))
print("Average Loss: " + str(round(loss, 5)))
print("Total Loss: " + str(round(loss*n, 5)))

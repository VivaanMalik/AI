import NeuralNetwork as nn
from mnist import MNIST # python-mnist
import cupy as cp

num_classes = 10
mnist_data = MNIST("samples")
images, labels = mnist_data.load_training()
images = [[j/255.0 for j in i] for i in images]

labels = nn.ConvertIntForClassification(labels, num_classes)
labels_modified = nn.AdjustOutput(labels, num_classes)

start_again = True
train = True
save = True

model = nn.NeuralNetwork()
if start_again:
    model.add(nn.Layer("Layer the first", 784, 256, nn.ReLU(),  nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the second", 256, 128, nn.ReLU(), nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the third", 128, 64, nn.ReLU(),   nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the fourth", 64, 10, None,        nn.Adam(learning_rate = 0.001), 0.0))

    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), nn.He(), nn.CosineAnnealing(0.001, 1e-5), nn.L2Regularization(1e-4))
else:
    model.load_data_from_JSON("data.json")
    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), WeightDecayFunc=nn.L2Regularization(1e-4))

if train:
    model.train(images, labels_modified, 10, 32)
if save:
    model.load_data_to_JSON("data.json")

print("\nEvaluation of Training Data")
evaluationdata = model.evaluate(images, labels)
acc = evaluationdata["Accuracy"]
cm = evaluationdata["ConfusionMatrix"]
print("Accuracy: "+str(round(acc*100, 2))+"%")
print(nn.PrettyPrintMatrix(cm))

print("\nEvaluation of Testing Data")
mnist_data = MNIST("samples")
images, labels = mnist_data.load_testing()
images = [[j/255.0 for j in i] for i in images]
labels = cp.eye(10)[labels].tolist()

evaluationdata = model.evaluate(images, labels)
acc = evaluationdata["Accuracy"]
cm = evaluationdata["ConfusionMatrix"]
print("Accuracy: "+str(round(acc*100, 2))+"%")
print(nn.PrettyPrintMatrix(cm))
import NeuralNetwork as nn
from mnist import MNIST
import cupy as cp

num_classes = 10
mnist_data = MNIST("samples")
images, labels = mnist_data.load_training()
images = [[j/255.0 for j in i] for i in images]
labels = cp.eye(num_classes)[labels].tolist()

epsilon = 0.01
labels_modified = [[epsilon if i == 0.0 else 1-(epsilon*(num_classes-1)) for i in j] for j in labels]

# images = cp.array(images)
# images = (images - cp.mean(images)) / cp.std(images)
# images = images.tolist()

start_again = True
train = True
save = True

model = nn.NeuralNetwork()
if start_again:
    model.add(nn.Layer("Layer the first", 784, 256, nn.ReLU(),  nn.Adam(learning_rate= 0.01), 0.2))
    model.add(nn.Layer("Layer the second", 256, 128, nn.ReLU(), nn.Adam(learning_rate= 0.01), 0.2))
    model.add(nn.Layer("Layer the third", 128, 64, nn.ReLU(),   nn.Adam(learning_rate= 0.01), 0.2))
    model.add(nn.Layer("Layer the fourth", 64, 10, None,        nn.Adam(learning_rate= 0.01), 0.0))

    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), nn.He(), nn.CosineAnnealing(0.0001, 1e-5))
else:
    model.load_data_from_JSON("data.json")
    model.compile_network(nn.SoftmaxCategoricalCrossEntropy())

if train:
    model.train(images, labels_modified, 20, 32)
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
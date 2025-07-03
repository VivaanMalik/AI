import NeuralNetwork as nn
import os
from mnist import MNIST # python-mnist'
from multiprocessing import Process, set_start_method

num_classes = 10
mnist_data = MNIST("samples")
images, labels = mnist_data.load_training()
images = [[j/255.0 for j in i] for i in images]

labels = nn.ConvertIntForClassification(labels, num_classes)
labels_modified = nn.AdjustOutput(labels, num_classes)

testimgs, testlabels = mnist_data.load_testing()
testimgs = [[j/255.0 for j in i] for i in testimgs]
testlabels = nn.ConvertIntForClassification(testlabels, num_classes)

def modeluse(i):
    model = nn.NeuralNetwork()
    model.add(nn.Layer("Layer the first", 784, 256, nn.ReLU(),  nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the second", 256, 128, nn.ReLU(), nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the third", 128, 64, nn.ReLU(),   nn.Adam(learning_rate = 0.001), 0.2))
    model.add(nn.Layer("Layer the fourth", 64, 10, None,        nn.Adam(learning_rate = 0.001), 0.0))
    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), nn.He(), nn.CosineAnnealing(0.001, 1e-5), nn.L2Regularization(1e-4))
    model.train(images, labels_modified, 10, 32)
    model.load_data_to_JSON("data\\data"+str(i)+".json")

    model = nn.NeuralNetwork()
    model.load_data_from_JSON("data\\data"+str(i)+".json")
    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), nn.He(), nn.CosineAnnealing(0.001, 1e-5), nn.L2Regularization(1e-4))

    evaluationdata = model.evaluate(testimgs, testlabels)
    acc = evaluationdata["Accuracy"]
    cm = evaluationdata["ConfusionMatrix"]
    print("Model: "+str(i)+"\nAccuracy: "+str(round(acc*100, 2))+"%\n" + nn.PrettyPrintMatrix(cm)+"\n\n")

num_models = 8
processes = []
if __name__ == "__main__":
    from multiprocessing import get_start_method
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    for i in range(num_models):
        p = Process(target=modeluse, args=(i,))
        p.start()
        processes.append(p)

    for i in range(num_models):
        processes[i].join()

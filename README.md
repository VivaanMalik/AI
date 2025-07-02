# Unnamed Machine Learning Framework (temp name)
---
## Table of Contents
<!-- TOC -->
## Features
#### Semisequential Setup
> [!WARNING]
>**Incoming yap**
> You can add layers to your network and compile them, but a few functions have to be defined as a parameter when the layer is defined.
> 
> The functions are not separated from the layer as I did not know about the "Sequential Setup" that is widely used until I tested my network on tensorflow to compare speeds and accuracy. 
>
>This is a subtle flex that the first ML Framework I used (even semiseriously) was my own.

#### Modifed Layers
Each Layer has its own
* unique custom ID
* Initialization function
* Activation function
* Optimization function
* dropout probability

#### Load data to and from a specified JSON file
This function can store and retrieve 
* No of Epoch the network has run for (including all previous runs)
* weights and biases
* custom IDs for each layer
* network layer configuration
* activation function per layer
* optimization function per layer 
* any paramters (if applicable) that the optimization function needs.
  
#### Training function
* Handles everything for you
* Can change batch size
* Auto shuffles datasets in batches
* Prints loss and time taken for every epoch

#### Evaluation function
Evaluates the network and send the following data
* Accuracy
* A label-wise confusion matrix that can be color coded using ```utils.py```

#### Initialization Functions
* Xavier
* He
* Uniform Xavier

#### Activation Functions
* Sigmoid
* ReLU
* Leaky ReLU
* Softmax*
    ⚠️ **Important**: While Softmax is indeed an activation function, it has been defined within ```loss.py``` as a combination of Softmax and CCE since they are often used together.

#### Loss Functions
* Binary Cross Entropy (BCE)
* Mean Squared Error (MSE)
* Softmax + Categorical Cross Entropy (CCE)

#### Learning Rate Decay Functions
* Step Decay
* Exponential Decay
* Linear Decay
* Cosine Annealing
  
#### Optimization Functions
* Stochastic Gradient Descent (SGD)
* SGD with momentum
* Nesterov Accelerated Gradient (NAG)
* Root Mean Square Propogation (RMSProp)
* Adaptive Movement Estimation (Adam)

#### Utility Functions

* Pretty Print Matrix
    💡 **Note:** This is designed specifically for the confusion matrix returned from ```NeuralNetwork().NeuralNetwork().evaluate()```.
* Adjust Output (normalization?)
* Convert Int For Classification (eg: 0 = [1, 0, ...], 1 = [0, 1, ...])
  
## Documentation of classes and methods (per file)

#### network
> class NeuralNetwork()
> `no parameters`

> method add(layer)
> `layer - Layer class to add`

> method load_data_from_JSON(filepath)
> `filepath - file path of JSON file to use`

> method load_data_to_JSON(filepath)
> `filepath - file path of JSON file to use`

> method compile_network(loss, initializer = None, lrdecayfunc = None)
> ```
> loss - set loss function
> initializer - set initializer
> lrdecayfunc - set decay function for learning rate
> ```

> method train(input_values, target_values, epochs, batch_size)
> ```
> input_values - input value dataset list
> target_values - target value dataset list
> epochs - no. of forward+baclward propogation cycles
> batch_size - affects how much the neural network 'spreads'
> ```

> method predict(input_values)
> ```
> input_values - input value dataset list
> ```

> method evaluate(testinput, testtargetoutput)
> ```
> testinput - input value dataset list
> testtargetoutput - target value dataset list
> ```

#### layer
> class Layer(ID, PreviousLayerNodeCount, CurrentLayerNodeCount, ActivationFunction = None, OptimizingFunction = StochasticGradientDescent(), DropOutProbability = 0)
> ```
> ID - Custom 'name' to give the layer
> PreviousLayerNodeCount - no. of nodes in the previous layer
> CurrentLayerNodeCount - no. of nodes in the layer being defined
> ActivationFunction - class of the activation function, if None its passed without being modified
> OptimizingFunction - class of the Optimization function, SGD by default
> DropOutProbability - probability to drop nodes out, 0 by default
> ```

#### initializer
> class Xavier()
> `no parameters`

> class He()
> `no parameters`

> class XavierUniform()
> `no parameters`

#### activations
> class Sigmoid()
> `no parameters`

> class ReLU()
> `no parameters`

> class LeakyReLU()
> `no parameters`

#### losses
> class BinaryCrossEntropy()
> `no parameters`

> class SoftmaxCategoricalCrossEntropy()
> `no parameters (use None for activation function)`

> class MeanSquaredError()
> `no parameters`

#### optimizer
> class StochasticGradientDescent(learning_rate = 0.01, load_param = None)
> ```
> learning_rate - affects change in weights and biases
> load_param - this does not need to be filled out
> ```

> class SGDMomentum(momentum_coeff = 0.9, learning_rate = 0.01, load_param = None)
> ```
> momentum_coeff - affects how much of the previous momentum is retained
> learning_rate - affects change in weights and biases
> load_param - this does not need to be filled out
> ```

> class NesterovAcceleratedGradient(momentum_coeff = 0.9, learning_rate = 0.01, load_param = None)
> ```
> momentum_coeff - affects how much of the previous momentum is retained
> learning_rate - affects change in weights and biases
> load_param - this does not need to be filled out
> ```

> class RMSProp(decay_rate = 0.9, learning_rate = 0.01, load_param = None)
> ```
> decay_rate - affects how fast the change in weights and biases change
> learning_rate - affects change in weights and biases
> load_param - this does not need to be filled out
> ```

> class Adam(first_moment_decay_rate = 0.9, second_moment_decay_rate = 0.999, learning_rate = 0.001, timestep = 0)
> ```
> first_moment_decay_rate - affects mean
> second_moment_decay_rate - affects variance
> learning_rate - affects change in weights and biases
> timestep - this does not need to be filled out
> load_param - this does not need to be filled out
> ```

#### learning_rate_decay
> class StepDecay(initial_lr, decay_step_size, decay_factor = 0.5)
> ```
> initial_lr - initial learning rate to start with
> decay_step_size - updates learning rate every _ steps
> decay_factor - affects how drastically learning rate is changed
> ```

> class ExponentialDecay(initial_lr, decay_constant = 0.01)
> ```
> initial_lr - initial learning rate to start with
> decay_constant - affects how drastically learning rate is changed
> ```

> class LinearDecay(initial_lr, total_epoch = None)
> ```
> initial_lr - initial learning rate to start with
> total_epoch - this does not need to be filled out
> ```

> class CosineAnnealing(initial_lr, min_lr, total_epoch = None)
> ```
> initial_lr - initial learning rate to start with
> min_lr - lower bound for the lerning rate
> total_epoch - this does not need to be filled out
> ```


## Example program
```
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
    model.add(nn.Layer("Layer the first", 784, 256, nn.ReLU(),  nn.Adam(learning_rate= 0.001), 0.2))
    model.add(nn.Layer("Layer the second", 256, 128, nn.ReLU(), nn.Adam(learning_rate= 0.001), 0.2))
    model.add(nn.Layer("Layer the third", 128, 64, nn.ReLU(),   nn.Adam(learning_rate= 0.001), 0.2))
    model.add(nn.Layer("Layer the fourth", 64, 10, None,        nn.Adam(learning_rate= 0.001), 0.0))

    model.compile_network(nn.SoftmaxCategoricalCrossEntropy(), nn.He(), nn.CosineAnnealing(0.001, 1e-5))
else:
    model.load_data_from_JSON("data.json")
    model.compile_network(nn.SoftmaxCategoricalCrossEntropy())

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
```
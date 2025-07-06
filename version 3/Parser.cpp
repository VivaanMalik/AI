#include "header.hpp"

json ParseAndComputeData(string description) {
    /*
        Here we now define the method of storage of information
        List of Arbitrary functions by category code
        Activation:     0000    
                Sigmoid         000
                ReLU            001
                LeakyReLU       010
        Intializer:     0001
                Xavier          000
                He              001
                UniformXavier   010
        LR Decay:       0010
                Step decay      000
                Exp decay       001
                linear decay    010
                cos annealing   011
        Loss:           0100
                BCE             000
                Softmax + CCR   001
                MSE             010
        Optimizer:      0101
                SGD             000
                SGDMomentum     001
                NAG             010
                RMSProp         011
                Adam            100
        weight decay:   0110
                L2regular..     000

        description boilerplate

        0000 000 pointers separated by | or somn
    */
    // TODO: everything here
    // sample data
    json result;
    result["name"] = "BingBong";
    return result;
}
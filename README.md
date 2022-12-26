## mllib

A machine learning library

NOTE: WORK IN PROGRESS!  An initial release should be ready in the coming weeks.

## Introduction

This library is intended to be a playground for learning a number of machine learning algorithms from "first principles."  Currently the library includes a simple neural network implementation.

## Neural Networks


### Building a neural network

Build a neural network using the builder interface:

```
        Network network = Network.builder()
                .numInputUnits(120)
                .layers(List.of(
                        new Layer(20, new Relu()),
                        new Layer(10, new Relu()),
                        new Layer(1, new Sigmoid())
                ))
                .costFunction(new MSE())
                .build();

```

In this example, we have a four layer network - 1 input layer, 2 hidden layers, and the output layer.  Note the input layer is implicit, and not included in the list of layers.  This network has 120 inputs and uses the "Mean Squared Error" cost function.  The first hidden layer has 20 outputs and uses the Rectified Linear Unit activation function.

Once the network is built, you need to initialize it:

```
        network.initialize();
```

This will initialize the weights of the network to a small random value in the interval [-0.5, 0.5].

### Training the network

TODO


### Utilizing the network

TODO


### To Do

The implementation still lacks a few basic features that will be added over the coming weeks

* Softmax activation function
* store and load network configuration, including weights
* support for parallel processing
* abstract away the EJML stuff so client doesn't need that import
* regularization - L2, possibly drop out
* support for other cost functions, i.e. cross-entropy
* support for other initializers, i.e. Xavier
* support for other optimizers, i.e. Adam

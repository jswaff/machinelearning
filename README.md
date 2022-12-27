## mllib

A machine learning library

NOTE: WORK IN PROGRESS!  

## Introduction

This library is intended to be a playground for learning a number of machine learning algorithms from "first principles."  Currently the library includes a simple neural network implementation.

## Neural Networks

Currently only fully connected networks are supported.  The optimizer uses gradient descent, either stochastic or (mini)batches.  The implementation is vectorized, meaning mini-batches are processed in one pass to take advantage of the capabilities of linear algebra libraries.  

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

In this example, we have a four layer network - 1 input layer, 2 hidden layers, and the output layer.  Note the input layer is implicit, and not included in the list of layers.  This network has 120 inputs and uses the "Mean Squared Error" cost function.  The first hidden layer has 20 neurons and uses the Rectified Linear Unit activation function.

Once the network is built, you need to initialize it:

```
        network.initialize();
```

This will initialize the weights of the network to a small random value in the interval [-0.5, 0.5].

### Training the network

TODO


### Utilizing the network

TODO


### MNIST

As a functional test, a net has been trained to predict the label of handwritten digits.  This is the "hello world" of deep neural networks.  Even in the early stages it's been able to achieve nearly 97%.  State of the art is very close to 100% though, so there's a ways to go!

```
executing mnist testapp

loaded 60000 training images
loaded 10000 test images
initial cost: 3.261900902283934
        cost(0): 0.12908789221381875
        cost(10): 0.06391633455139145
        cost(20): 0.05746017937420813
final cost: 0.05658565036013813
accuracy: 9676 / 10000 (96.76%)

execution complete.  bye.
```

### To Do

* additional activation functions, i.e. tanh, softmax
* abstract away the EJML (linear algebra library) stuff from the interface
* early stopping criteria
* store and load network configuration
* regularization - L2, possibly drop out
* support for other cost functions, i.e. cross-entropy
* support for other initializers, i.e. Xavier
* support for other optimizers, i.e. Adam
* convolutional nets
* recurrent nets
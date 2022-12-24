package com.jamesswafford.ml.nn.activation;

public class Sigmoid implements ActivationFunction {

    @Override
    public Double func(Double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    @Override
    public Double derivativeFunc(Double z) {
        double x = func(z);
        return x * (1.0 - x);
    }
}

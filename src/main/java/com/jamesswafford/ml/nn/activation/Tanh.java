package com.jamesswafford.ml.nn.activation;

public class Tanh implements ActivationFunction {
    @Override
    public Double func(Double z) {
        return Math.tanh(z);
    }

    @Override
    public Double derivativeFunc(Double a) {
        double x = Math.tanh(a);
        return 1 - (x * x);
    }
}

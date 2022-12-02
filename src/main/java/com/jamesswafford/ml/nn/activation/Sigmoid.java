package com.jamesswafford.ml.nn.activation;

public class Sigmoid implements ActivationFunction {

    @Override
    public Double a(Double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    @Override
    public Double dA(Double z) {
        double x = a(z);
        return x * (1 - x);
    }
}

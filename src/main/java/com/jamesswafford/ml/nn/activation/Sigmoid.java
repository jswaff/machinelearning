package com.jamesswafford.ml.nn.activation;

public class Sigmoid implements ActivationFunction {

    public static Sigmoid INSTANCE = new Sigmoid();

    private Sigmoid() {
    }

    @Override
    public String getName() {
        return "sigmoid";
    }

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

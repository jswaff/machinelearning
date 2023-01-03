package com.jamesswafford.ml.nn.activation;

public class Tanh implements ActivationFunction {

    public static Tanh INSTANCE = new Tanh();

    private Tanh() {
    }

    @Override
    public String getName() {
        return "tanh";
    }

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

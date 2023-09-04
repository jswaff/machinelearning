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
    public double func(double z) {
        return Math.tanh(z);
    }

    @Override
    public double derivativeFunc(double a) {
        double x = Math.tanh(a);
        return 1 - (x * x);
    }
}

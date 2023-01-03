package com.jamesswafford.ml.nn.activation;

public class Relu implements ActivationFunction {

    public static Relu INSTANCE = new Relu();

    private Relu() {
    }

    @Override
    public String getName() {
        return "relu";
    }

    @Override
    public Double func(Double z) {
        return Math.max(z, 0.0);
    }

    @Override
    public Double derivativeFunc(Double a) {
        return a < 0.0 ? 0.0 : 1.0;
    }
}

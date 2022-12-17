package com.jamesswafford.ml.nn.activation;

public class Relu implements ActivationFunction {

    @Override
    public Double func(Double z) {
        return Math.max(z, 0.0);
    }

    @Override
    public Double derivativeFunc(Double a) {
        return a < 0.0 ? 0.0 : 1.0;
    }
}

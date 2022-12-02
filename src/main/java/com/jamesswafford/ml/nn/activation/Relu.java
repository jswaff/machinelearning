package com.jamesswafford.ml.nn.activation;

public class Relu implements ActivationFunction {

    @Override
    public Double a(Double z) {
        return Math.max(z, 0.0);
    }

    @Override
    public Double dA(Double a) {
        return a < 0.0 ? 0.0 : 1.0;
    }
}

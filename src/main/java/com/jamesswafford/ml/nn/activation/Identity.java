package com.jamesswafford.ml.nn.activation;

public class Identity implements ActivationFunction {

    @Override
    public Double a(Double z) {
        return z;
    }

    @Override
    public Double dA(Double a) {
        return 1.0;
    }
}

package com.jamesswafford.ml.nn.activation;

public class Identity implements ActivationFunction {

    @Override
    public Double func(Double z) {
        return z;
    }

    @Override
    public Double derivativeFunc(Double a) {
        return 1.0;
    }
}

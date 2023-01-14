package com.jamesswafford.ml.nn.activation;

public class Identity implements ActivationFunction {

    public static Identity INSTANCE = new Identity();

    private Identity() {
    }

    @Override
    public String getName() {
        return "identity";
    }

    @Override
    public Double func(Double z) {
        return z;
    }

    @Override
    public Double derivativeFunc(Double a) {
        return 1.0;
    }
}

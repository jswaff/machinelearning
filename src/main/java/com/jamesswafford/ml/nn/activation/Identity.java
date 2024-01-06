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
    public double func(double z) {
        return z;
    }

    @Override
    public double derivativeFunc(double a) {
        return 1.0;
    }
}

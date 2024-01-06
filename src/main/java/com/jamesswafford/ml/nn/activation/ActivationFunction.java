package com.jamesswafford.ml.nn.activation;

// TODO: vectorized methods
public interface ActivationFunction {

    String getName();

    double func(double z);

    double derivativeFunc(double a);

}

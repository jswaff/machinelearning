package com.jamesswafford.ml.nn.activation;

public interface ActivationFunction {

    String getName();

    double func(double z);

    double derivativeFunc(double a);

}

package com.jamesswafford.ml.nn.activation;

// TODO: change interface so function is applied to a matrix
public interface ActivationFunction {

    String getName();

    Double func(Double z);

    Double derivativeFunc(Double a);

}

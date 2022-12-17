package com.jamesswafford.ml.nn.activation;

public interface ActivationFunction {

    Double func(Double z);

    Double derivativeFunc(Double a);

}

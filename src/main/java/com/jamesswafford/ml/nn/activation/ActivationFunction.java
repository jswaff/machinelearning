package com.jamesswafford.ml.nn.activation;

public interface ActivationFunction {

    Double a(Double z);

    Double dA(Double a);

}

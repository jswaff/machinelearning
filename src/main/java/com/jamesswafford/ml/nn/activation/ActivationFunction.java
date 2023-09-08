package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActivationFunction {

    String getName();

    double func(double z);

    INDArray func(INDArray z);

    double derivativeFunc(double a);

    INDArray derivativeFunc(INDArray a);

}

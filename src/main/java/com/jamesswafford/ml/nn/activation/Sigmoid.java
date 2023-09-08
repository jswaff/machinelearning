package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Sigmoid implements ActivationFunction {

    public static Sigmoid INSTANCE = new Sigmoid();

    private Sigmoid() {
    }

    @Override
    public String getName() {
        return "sigmoid";
    }

    @Override
    public double func(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    @Override
    public INDArray func(INDArray z) {
        return Transforms.sigmoid(z);
    }

    @Override
    public double derivativeFunc(double z) {
        double x = func(z);
        return x * (1.0 - x);
    }

    @Override
    public INDArray derivativeFunc(INDArray a) {
        return Transforms.sigmoidDerivative(a);
    }
}

package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Tanh implements ActivationFunction {

    public static Tanh INSTANCE = new Tanh();

    private Tanh() {
    }

    @Override
    public String getName() {
        return "tanh";
    }

    @Override
    public double func(double z) {
        return Math.tanh(z);
    }

    @Override
    public INDArray func(INDArray z, boolean copy) {
        return Transforms.hardTanh(z, copy);
    }

    @Override
    public double derivativeFunc(double a) {
        double x = Math.tanh(a);
        return 1 - (x * x);
    }

    @Override
    public INDArray derivativeFunc(INDArray a, boolean copy) {
        return Transforms.hardTanhDerivative(a, copy);
    }
}

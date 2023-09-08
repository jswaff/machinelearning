package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Relu implements ActivationFunction {

    public static Relu INSTANCE = new Relu();

    private Relu() {
    }

    @Override
    public String getName() {
        return "relu";
    }

    @Override
    public double func(double z) {
        return Math.max(z, 0.0);
    }

    @Override
    public INDArray func(INDArray z) {
        return Transforms.leakyRelu(z, 0.01);
    }

    @Override
    public double derivativeFunc(double a) {
        return a < 0.0 ? 0.0 : 1.0;
    }

    @Override
    public INDArray derivativeFunc(INDArray a) {
        return Transforms.leakyReluDerivative(a, 0.01);
    }
}

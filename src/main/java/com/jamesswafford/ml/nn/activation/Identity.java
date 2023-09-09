package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
    public INDArray func(INDArray z, boolean copy) {
        return copy ? z.dup() : z;
    }

    @Override
    public double derivativeFunc(double a) {
        return 1.0;
    }

    @Override
    public INDArray derivativeFunc(INDArray a, boolean copy) {
        return Nd4j.ones(a.dataType(), a.rows(), a.columns());
    }
}

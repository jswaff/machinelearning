package com.jamesswafford.ml.nn.cost;

public class CostFunctionFactory {

    public static CostFunction create(String costFunctionName) {

        if ("mse".equalsIgnoreCase(costFunctionName)) return MSE.INSTANCE;

        throw new IllegalArgumentException("Don't know how to create cost function: " + costFunctionName);
    }
}

package com.jamesswafford.ml.nn.activation;

public class ActivationFunctionFactory {

    public static ActivationFunction create(String functionName) {

        if ("identity".equalsIgnoreCase(functionName)) return Identity.INSTANCE;
        if ("relu".equalsIgnoreCase(functionName)) return Relu.INSTANCE;
        if ("sigmoid".equalsIgnoreCase(functionName)) return Sigmoid.INSTANCE;
        if ("tanh".equalsIgnoreCase(functionName)) return Tanh.INSTANCE;

        throw new IllegalArgumentException("Don't know how to create activation function: " + functionName);
    }

}

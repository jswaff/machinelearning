package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ReluTests {

    private final Relu relu = new Relu();

    @Test
    void activation() {
        assertEquals(5.0, relu.func(5.0));
        assertEquals(0.0, relu.func(0.0));
        assertEquals(0.0, relu.func(-1.0));
    }

    @Test
    void derivative() {
        assertEquals(0.0, relu.derivativeFunc(-0.00001));
        assertEquals(1.0, relu.derivativeFunc(0.0)); // technically undefined
        assertEquals(1.0, relu.derivativeFunc(0.00001));
    }
}

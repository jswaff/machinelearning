package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;


public class TanhTests {

    private final Tanh tanh = new Tanh();

    @Test
    void activation() {
        assertDoubleEquals(0.0, tanh.func(0.0));
        assertDoubleEquals(0.46211715726, tanh.func(0.5));
        assertDoubleEquals(-0.46211715726, tanh.func(-0.5));

        assertDoubleEquals(0.76159415595, tanh.func(1.0));
        assertDoubleEquals(-0.76159415595, tanh.func(-1.0));
    }

    @Test
    void derivative() {
        assertDoubleEquals(1.0, tanh.derivativeFunc(0.0));

        assertDoubleEquals(0.41997434161402614, tanh.derivativeFunc(1.0));
        assertDoubleEquals(0.41997434161402614, tanh.derivativeFunc(-1.0));

        assertDoubleEquals(0.07065082485316443, tanh.derivativeFunc(2.0));
        assertDoubleEquals(0.07065082485316443, tanh.derivativeFunc(-2.0));
    }
}

package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;


public class TanhTests {
    private static final double epsilon = 0.00001;
    private final Tanh tanh = Tanh.INSTANCE;

    @Test
    void activation() {
        assertEquals(0.0, tanh.func(0.0), epsilon);
        assertEquals(0.46211715726, tanh.func(0.5), epsilon);
        assertEquals(-0.46211715726, tanh.func(-0.5), epsilon);

        assertEquals(0.76159415595, tanh.func(1.0), epsilon);
        assertEquals(-0.76159415595, tanh.func(-1.0), epsilon);
    }

    @Test
    void derivative() {
        assertEquals(1.0, tanh.derivativeFunc(0.0));

        assertEquals(0.41997434161402614, tanh.derivativeFunc(1.0), epsilon);
        assertEquals(0.41997434161402614, tanh.derivativeFunc(-1.0), epsilon);

        assertEquals(0.07065082485316443, tanh.derivativeFunc(2.0), epsilon);
        assertEquals(0.07065082485316443, tanh.derivativeFunc(-2.0), epsilon);
    }
}

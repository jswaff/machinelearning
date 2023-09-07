package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SigmoidTests {

    private static final double epsilon = 0.00001;
    private final Sigmoid sigmoid = Sigmoid.INSTANCE;

    @Test
    void activation() {
        assertEquals(0.5, sigmoid.func(0.0));

        assertEquals(0.6224593, sigmoid.func(0.5), epsilon);
        assertEquals(0.3775407, sigmoid.func(-0.5), epsilon);

        assertEquals(0.7310586, sigmoid.func(1.0), epsilon);
        assertEquals(0.2689414, sigmoid.func(-1.0), epsilon);
    }

    @Test
    void derivative() {
        assertEquals(0.25, sigmoid.derivativeFunc(0.0));

        assertEquals(0.2350037, sigmoid.derivativeFunc(0.5), epsilon);
        assertEquals(0.2350037, sigmoid.derivativeFunc(-0.5), epsilon);

        assertEquals(0.1966119, sigmoid.derivativeFunc(1.0), epsilon);
        assertEquals(0.1966119, sigmoid.derivativeFunc(-1.0), epsilon);
    }

}

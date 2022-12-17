package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class SigmoidTests {

    private final Sigmoid sigmoid = new Sigmoid();

    @Test
    void activation() {
        assertDoubleEquals(0.5, sigmoid.func(0.0));

        assertDoubleEquals(0.6224593, sigmoid.func(0.5));
        assertDoubleEquals(0.3775407, sigmoid.func(-0.5));

        assertDoubleEquals(0.7310586, sigmoid.func(1.0));
        assertDoubleEquals(0.2689414, sigmoid.func(-1.0));
    }

    @Test
    void derivative() {
        assertDoubleEquals(0.25, sigmoid.derivativeFunc(0.0));

        assertDoubleEquals(0.2350037, sigmoid.derivativeFunc(0.5));
        assertDoubleEquals(0.2350037, sigmoid.derivativeFunc(-0.5));

        assertDoubleEquals(0.1966119, sigmoid.derivativeFunc(1.0));
        assertDoubleEquals(0.1966119, sigmoid.derivativeFunc(-1.0));
    }

}

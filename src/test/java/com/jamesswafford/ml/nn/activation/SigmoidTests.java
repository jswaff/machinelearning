package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class SigmoidTests {

    private final Sigmoid sigmoid = new Sigmoid();

    @Test
    void activation() {
        assertDoubleEquals(0.5, sigmoid.a(0.0));

        assertDoubleEquals(0.6224593, sigmoid.a(0.5));
        assertDoubleEquals(0.3775407, sigmoid.a(-0.5));

        assertDoubleEquals(0.7310586, sigmoid.a(1.0));
        assertDoubleEquals(0.2689414, sigmoid.a(-1.0));
    }

    @Test
    void derivative() {
        assertDoubleEquals(0.25, sigmoid.dA(0.0));

        assertDoubleEquals(0.2350037, sigmoid.dA(0.5));
        assertDoubleEquals(0.2350037, sigmoid.dA(-0.5));

        assertDoubleEquals(0.1966119, sigmoid.dA(1.0));
        assertDoubleEquals(0.1966119, sigmoid.dA(-1.0));
    }

}

package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.Identity;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class LayerTests {

    @Test
    public void initialize() {
        Layer layer = new Layer(10, new Identity());
        layer.initialize(3);

        // biases are initialized to 0
        assertEquals(0.0, layer.getBias(0));
        assertEquals(0.0, layer.getBias(4));
        assertEquals(0.0, layer.getBias(9));

        // weights are initialized to a small random number between 0 and 1
        assertTrue(layer.getWeight(0, 0) > 0.0);
        assertTrue(layer.getWeight(0, 0) < 1.0);
    }
}

package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.Identity;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;
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
        for (int i=0;i<10;i++) {
            for (int j=0;j<3;j++) {
                assertTrue(layer.getWeight(i, j) > 0.0);
                assertTrue(layer.getWeight(i, j) < 1.0);
            }
        }
    }

    @Test
    public void linearForwardSingleUnit() {
        Layer layer = new Layer(1, new Identity());
        layer.initialize(1);
        layer.setWeight(0, 0, 0.1);
        layer.setBias(0, 0.05);

        SimpleMatrix X = new SimpleMatrix(1, 1);
        X.set(0, 0, 0.5);
        SimpleMatrix Z = layer.linearForward(X);

        assertEquals(1, Z.numRows());
        assertEquals(1, Z.numCols());
        assertEquals(0.1, Z.get(0, 0));
    }

    @Test
    public void linearForward1x3() {
        Layer layer = new Layer(3, new Identity());
        layer.initialize(1);
        layer.setWeight(0, 0, 0.1);
        layer.setBias(0, 0.05);
        layer.setWeight(1, 0, 0.2);
        layer.setBias(1, 0.05);
        layer.setWeight(2, 0, 0.3);
        layer.setBias(2, 0.05);

        SimpleMatrix X = new SimpleMatrix(1, 1);
        X.set(0, 0, 2.0);
        SimpleMatrix Z = layer.linearForward(X);

        assertEquals(3, Z.numRows());
        assertEquals(1, Z.numCols());
        assertEquals(0.25, Z.get(0, 0));
        assertEquals(0.45, Z.get(1, 0));
        assertEquals(0.65, Z.get(2, 0));
    }

    @Test
    public void linearForward3x4() {
        Layer layer = new Layer(4, new Identity());
        layer.initialize(3);

        // unit 1 weights
        layer.setWeight(0, 0, 0.5);
        layer.setWeight(0, 1, 0.3);
        layer.setWeight(0, 2, 0.1);

        // unit 2 weights
        layer.setWeight(1, 0, 0.9);
        layer.setWeight(1, 1, 0.5);
        layer.setWeight(1, 2, 0.65);

        // unit 3 weights
        layer.setWeight(2, 0, 1.2);
        layer.setWeight(2, 1, 0.2);
        layer.setWeight(2, 2, -0.3);

        // unit 4 weights
        layer.setWeight(3, 0, 0.15);
        layer.setWeight(3, 1, 0.4);
        layer.setWeight(3, 2, 0.4);

        // bias
        for (int i=0;i<4;i++) {
            layer.setBias(i, 0.05);
        }

        SimpleMatrix X = new SimpleMatrix(3, 1);
        X.set(0, 0, 0.1);
        X.set(1, 0, 0.3);
        X.set(2, 0, -0.2);
        SimpleMatrix Z = layer.linearForward(X);

        assertEquals(4, Z.numRows());
        assertEquals(1, Z.numCols());
        assertDoubleEquals(0.17, Z.get(0, 0));
        assertDoubleEquals(0.16, Z.get(1, 0));
        assertDoubleEquals(0.29, Z.get(2, 0));
        assertDoubleEquals(0.105, Z.get(3, 0));
    }
}

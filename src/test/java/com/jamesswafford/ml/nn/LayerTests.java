package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import com.jamesswafford.ml.nn.activation.Identity;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class LayerTests {

    @Test
    public void initialize() {
        Layer layer = new Layer(10, new Identity());
        layer.initialize(3);

        // biases are initialized to 0
        for (int j=0;j<10;j++) {
            assertEquals(0.0, layer.getBias(j));
        }

        // weights are initialized to a small random number between 0 and 1
        for (int j=0;j<10;j++) {
            for (int k=0;k<3;k++) {
                assertTrue(layer.getWeight(j, k) > 0.0);
                assertTrue(layer.getWeight(j, k) < 1.0);
            }
        }
    }

    @Test
    public void forwardSingleUnit() {
        ActivationFunction activationFunction = Mockito.mock(ActivationFunction.class);
        Layer layer = new Layer(1, activationFunction);
        layer.initialize(1);
        layer.setWeight(0, 0, 0.1);
        layer.setBias(0, 0.05);

        SimpleMatrix X = new SimpleMatrix(1, 1);
        X.set(0, 0, 0.5);
        SimpleMatrix Z = layer.linearForward(X);

        assertEquals(1, Z.numRows());
        assertEquals(1, Z.numCols());
        assertEquals(0.1, Z.get(0, 0));

        layer.activationForward(X);
        verify(activationFunction, times(1)).a(0.1);
    }

    @Test
    public void forward1x3() {
        ActivationFunction activationFunction = Mockito.mock(ActivationFunction.class);
        Layer layer = new Layer(3, activationFunction);
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

        layer.activationForward(X);
        verify(activationFunction, times(1)).a(0.25);
        verify(activationFunction, times(1)).a(0.45);
        verify(activationFunction, times(1)).a(0.65);
    }

    ActivationFunction aFunc = new ActivationFunction() {
        @Override
        public Double a(Double z) {
            return z * 2;
        }

        @Override
        public Double dA(Double a) {
            return 2.0;
        }
    };

    @Test
    public void forward3x4_x1() {
        Layer layer = build3x4Layer(aFunc);

        // input a column vector (one row per unit from previous layer)
        SimpleMatrix X = new SimpleMatrix(3, 1);
        X.set(0, 0, 0.1);
        X.set(1, 0, 0.3);
        X.set(2, 0, -0.2);

        SimpleMatrix Z = layer.linearForward(X);

        // the output should be a column vector with one row per unit in this layer
        assertEquals(4, Z.numRows());
        assertEquals(1, Z.numCols());
        assertDoubleEquals(0.17, Z.get(0, 0));
        assertDoubleEquals(0.16, Z.get(1, 0));
        assertDoubleEquals(0.29, Z.get(2, 0));
        assertDoubleEquals(0.105, Z.get(3, 0));

        Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.activationForward(X);
        assertMatrixEquals(Z, Z_A.getValue0());
        SimpleMatrix A = Z_A.getValue1();
        assertEquals(4, A.numRows());
        assertEquals(1, A.numCols());
        assertDoubleEquals(0.17*2, A.get(0, 0));
        assertDoubleEquals(0.16*2, A.get(1, 0));
        assertDoubleEquals(0.29*2, A.get(2, 0));
        assertDoubleEquals(0.105*2, A.get(3, 0));
    }

    @Test
    // same layer, different inputs
    public void forward3x4_x2() {
        Layer layer = build3x4Layer(aFunc);

        // input a column vector (one row per unit from previous layer)
        SimpleMatrix X = new SimpleMatrix(3, 1);
        X.set(0, 0, 0.4);
        X.set(1, 0, -0.1);
        X.set(2, 0, 0.0);

        SimpleMatrix Z = layer.linearForward(X);

        // the output should be a column vector with one row per unit in this layer
        assertEquals(4, Z.numRows());
        assertEquals(1, Z.numCols());

        assertDoubleEquals(0.22, Z.get(0, 0));
        assertDoubleEquals(0.36, Z.get(1, 0));
        assertDoubleEquals(0.51, Z.get(2, 0));
        assertDoubleEquals(0.07, Z.get(3, 0));

        Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.activationForward(X);
        assertMatrixEquals(Z, Z_A.getValue0());
        SimpleMatrix A = Z_A.getValue1();
        assertEquals(4, A.numRows());
        assertEquals(1, A.numCols());
        assertDoubleEquals(0.22*2, A.get(0, 0));
        assertDoubleEquals(0.36*2, A.get(1, 0));
        assertDoubleEquals(0.51*2, A.get(2, 0));
        assertDoubleEquals(0.07*2, A.get(3, 0));
    }

    @Test
    void forward3x4_vectorized() {
        Layer layer = build3x4Layer(aFunc);

        SimpleMatrix X = new SimpleMatrix(3, 2);
        // from x1 test
        X.set(0, 0, 0.1);
        X.set(1, 0, 0.3);
        X.set(2, 0, -0.2);
        // from x2 test
        X.set(0, 1, 0.4);
        X.set(1, 1, -0.1);
        X.set(2, 1, 0.0);

        SimpleMatrix Z = layer.linearForward(X);

        // the output should have one row per unit and one column per training example
        assertEquals(4, Z.numRows());
        assertEquals(2, Z.numCols());

        // test x1
        assertDoubleEquals(0.17, Z.get(0, 0));
        assertDoubleEquals(0.16, Z.get(1, 0));
        assertDoubleEquals(0.29, Z.get(2, 0));
        assertDoubleEquals(0.105, Z.get(3, 0));

        // test x2
        assertDoubleEquals(0.22, Z.get(0, 1));
        assertDoubleEquals(0.36, Z.get(1, 1));
        assertDoubleEquals(0.51, Z.get(2, 1));
        assertDoubleEquals(0.07, Z.get(3, 1));

        Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.activationForward(X);
        assertMatrixEquals(Z, Z_A.getValue0());
        SimpleMatrix A = Z_A.getValue1();
        assertEquals(4, A.numRows());
        assertEquals(2, A.numCols());
        assertDoubleEquals(0.17*2, A.get(0, 0));
        assertDoubleEquals(0.16*2, A.get(1, 0));
        assertDoubleEquals(0.29*2, A.get(2, 0));
        assertDoubleEquals(0.105*2, A.get(3, 0));

        assertDoubleEquals(0.22*2, A.get(0, 1));
        assertDoubleEquals(0.36*2, A.get(1, 1));
        assertDoubleEquals(0.51*2, A.get(2, 1));
        assertDoubleEquals(0.07*2, A.get(3, 1));

    }

    private Layer build3x4Layer(ActivationFunction activationFunction) {
        Layer layer = new Layer(4, activationFunction);
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

        return layer;
    }
}

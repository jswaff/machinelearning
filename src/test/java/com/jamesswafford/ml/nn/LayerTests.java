package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import com.jamesswafford.ml.nn.activation.Identity;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Triplet;
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
    public void forwardAndBack_3x4_example1() {
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

        // backprop
        SimpleMatrix dA_l = new SimpleMatrix(4, 1, true, new double[] { 0, 1, 2, 3 }); // completely contrived
        Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> dA_dW_db = layer.backProp(dA_l, Z, X);

        // the gradients for the previous layer should be a 3x1 row vector
        SimpleMatrix dA = dA_dW_db.getValue0();
        assertEquals(3, dA.numRows());
        assertEquals(1, dA.numCols());

        assertDoubleEquals(7.5, dA.get(0, 0));
        assertDoubleEquals(4.2, dA.get(1, 0));
        assertDoubleEquals(2.5, dA.get(2, 0));

        // the delta weights should be the same shape as the weights matrix
        SimpleMatrix dW = dA_dW_db.getValue1();
        assertEquals(4, dW.numRows());
        assertEquals(3, dW.numCols());
        // dZ = dA_1 doubled, since the derivative of the activation function is always 2
        //    = [ 0 2 4 6 ]  (col vector  4 x 1)
        // A_prev.T = X.T = [ .1, .3, -.2 ]  (row vector 1 x 3)
        /* dW = dZ * A_prev.T
         0           0           0
        .2          .6         -.4
        .4          1.2        -.8
        .6          1.8        -1.2
        */
        assertDoubleEquals(0.6, dW.get(1, 1));
        assertDoubleEquals(-1.2, dW.get(3, 2));

        // db should be the same as dZ
        SimpleMatrix db = dA_dW_db.getValue2();
        assertEquals(4, db.numRows());
        assertEquals(1, db.numCols());
        assertDoubleEquals(2, db.get(1, 0));
        assertDoubleEquals(6, db.get(3, 0));
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

    @Test
    void updateWeightsAndBiases() {
        Layer layer = build3x4Layer(aFunc);

        SimpleMatrix dW = new SimpleMatrix(4, 3, true,
                new double[]{0.1, 0.1, 0.1,
                             0.2, 0.2, 0.2,
                             0.3, 0.3, 0.3,
                             0.4, 0.4, 0.4 });

        SimpleMatrix dB = new SimpleMatrix(4, 1, true,
                new double[] { 0.06, 0.06, 0.06, 0.06} );

        double w_0_0 = layer.getWeight(0, 0);
        double w_1_1 = layer.getWeight(1, 1);
        double w_2_2 = layer.getWeight(2, 2);

        double b_0 = layer.getBias(0);

        layer.updateWeightsAndBias(dW, dB);

        assertDoubleEquals(w_0_0 - 0.1, layer.getWeight(0, 0));
        assertDoubleEquals(w_1_1 - 0.2, layer.getWeight(1, 1));
        assertDoubleEquals(w_2_2 - 0.3, layer.getWeight(2, 2));

        assertDoubleEquals(b_0 - 0.06, layer.getBias(0));
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

package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Triplet;

@RequiredArgsConstructor
public class Layer {

    @Getter
    private final int numUnits;
    private final ActivationFunction activationFunction;

    private SimpleMatrix w;  // weights vector, j x k where j = units this layer, k = prev. layer
    private SimpleMatrix b;  // bias column vector, j x 1


    /**
     * Initialize this layer of the network by initializing the weights to small random values and
     * the biases to 0.
     *
     * @param numUnitsPreviousLayer the number of units in the previous layer
     */
    public void initialize(int numUnitsPreviousLayer) {
        w = new SimpleMatrix(numUnits, numUnitsPreviousLayer);
        for (int r=0;r<numUnits;r++) {
            for (int c=0;c<numUnitsPreviousLayer;c++) {
                w.set(r, c, Math.random());
            }
        }
        b = new SimpleMatrix(numUnits, 1);
        for (int r=0;r<numUnits;r++) {
            b.set(r, 0, 0.0);
        }
    }

    public Double getWeight(int unit, int prevUnit) {
        return w.get(unit, prevUnit);
    }

    public void setWeight(int unit, int prevUnit, Double val) {
        w.set(unit, prevUnit, val);
    }

    public Double getBias(int unit) {
        return b.get(unit, 0);
    }

    public void setBias(int unit, Double val) {
        b.set(unit, 0, val);
    }

    /**
     * Compute the linear portion (pre-activation) of the forward pass.
     *
     * @param prevA the activations matrix from the previous layer, of shape l_prev x m, where l_prev is the
     *              number of units in the previous layer, and m is the number of training examples.
     *
     * @return the Z matrix containing linear computation portion of the feed forward pass, of shape l x m,
     *              where l is the number of units in this layer, and m is the number of training examples.
     */
    public SimpleMatrix linearForward(SimpleMatrix prevA) {
        // there is no "broadcast" operator in the OO interface of the EJML lib.
        // if there were (column) broadcasting we could do something like w.mult(prevA).bcPlus(b);

        SimpleMatrix Z = w.mult(prevA);
        for (int m=0;m<Z.numCols();m++) {
            Z.setColumn(m, 0, Z.extractVector(false, m).plus(b).getDDRM().getData());
        }
        return Z;
    }

    /**
     * Compute the output (activation) of the forward pass.
     *
     * @param prevA the activations matrix from the previous layer, of shape l_prev x m, where l_prev is the
     *              number of units in the previous layer, and m is the number of training examples.
     *
     * @return the Z, A matrices containing the linear computation and activations of the feed forward pass.
     *         Each matrix has shape l x m, where l is the number of units in this layer, and m is the number of
     *         training examples.
     */
    public Pair<SimpleMatrix, SimpleMatrix> activationForward(SimpleMatrix prevA) {
        SimpleMatrix Z = linearForward(prevA);
        SimpleMatrix A = new SimpleMatrix(Z.numRows(), Z.numCols());
        for (int r=0;r<A.numRows();r++) {
            for (int c=0;c<A.numCols();c++) {
                A.set(r, c, activationFunction.a(Z.get(r, c)));
            }
        }
        return new Pair<>(Z, A);
    }

    /**
     * Perform the backprop step using gradient descent.
     * Note this step does NOT update weights and biases.
     *
     * @param dZ the gradients from the next layer.
     *
     * @return the gradients dZ, dW, db for this layer
     */
    public Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> backProp(SimpleMatrix dZ) {
        return new Triplet<>(dZ, dZ, dZ);
    }

    /**
     * Update weights and biases by subtracting dW, dB.
     *
     * @param dW - deltas for weights
     * @param db - deltas for biases
     */
    public void updateWeightsAndBias(SimpleMatrix dW, SimpleMatrix db) {
        w = w.minus(dW);
        b = b.minus(db);
    }
}

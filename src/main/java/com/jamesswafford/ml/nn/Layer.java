package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Random;

@RequiredArgsConstructor
public class Layer {

    @Getter
    private final int numUnits;
    private final ActivationFunction activationFunction;

    private SimpleMatrix w;  // weights vector, j x k where j = units this layer, k = prev. layer
    private SimpleMatrix b;  // bias column vector, j x 1

    private SimpleMatrix X;  // input from previous layer
    private SimpleMatrix Z;  // the linear computation portion of the output
    private SimpleMatrix A;  // output of this layer -- g(Z)

    /**
     * Initialize this layer of the network by initializing the weights to small random values and
     * the biases to 0.
     *
     * @param numUnitsPreviousLayer the number of units in the previous layer
     */
    public void initialize(int numUnitsPreviousLayer) {
        initialize(numUnitsPreviousLayer, 0);
    }

    public void initialize(int numUnitsPreviousLayer, long seed) {
        Random rand = new Random(seed);
        w = new SimpleMatrix(numUnits, numUnitsPreviousLayer);
        for (int r=0;r<numUnits;r++) {
            for (int c=0;c<numUnitsPreviousLayer;c++) {
                w.set(r, c, rand.nextDouble());
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

    public SimpleMatrix getWeights() { return w; }

    public void setWeight(int unit, int prevUnit, Double val) {
        w.set(unit, prevUnit, val);
    }

    public Double getBias(int unit) {
        return b.get(unit, 0);
    }

    public void setBias(int unit, Double val) {
        b.set(unit, 0, val);
    }

    public SimpleMatrix getZPrime() {
        SimpleMatrix Z_prime = new SimpleMatrix(Z.numRows(), Z.numCols());
        // unfortunately no broadcast operator
        for (int r=0;r<Z_prime.numRows();r++) {
            for (int c=0;c<Z_prime.numCols();c++) {
                Z_prime.set(r, c, activationFunction.derivativeFunc(Z.get(r, c)));
            }
        }
        return Z_prime;
    }

    /**
     * Perform the forward computation step.  The output is the pair <Z, A>, where Z is the linear portion of the
     * computation and A is the activation function applied to Z.
     *
     * @param X the inputs from the previous layer, of shape n x m, where n is the number of units in the previous
     *          layer, and m is the number of training examples.
     *
     * @return the Z, A matrices containing the linear computation and activations of the feed forward pass.
     *         Each matrix has shape l x m, where l is the number of units in this layer, and m is the number of
     *         training examples.
     */
    public Pair<SimpleMatrix, SimpleMatrix> feedForward(SimpleMatrix X) {
        this.X = X;

        // there is no "broadcast" operator in the OO interface of the EJML lib.
        // if there were (column) broadcasting we could do something like w.mult(X).bcPlus(b);
        Z = w.mult(X);
        for (int m=0;m<Z.numCols();m++) {
            Z.setColumn(m, 0, Z.extractVector(false, m).plus(b).getDDRM().getData());
        }

        A = new SimpleMatrix(Z.numRows(), Z.numCols());
        for (int r=0;r<A.numRows();r++) {
            for (int c=0;c<A.numCols();c++) {
                A.set(r, c, activationFunction.func(Z.get(r, c)));
            }
        }
        return new Pair<>(Z, A);
    }

    /**
     * Perform the backwards propagation step using gradient descent.
     * Note this step does NOT update weights and biases.
     *
     * @param A_error the error of the activation function
     *
     * @return dW, db
     */
    public Pair<SimpleMatrix, SimpleMatrix> backProp(SimpleMatrix A_error) {

        // the adjustment to the weights is proportional to how active the feature was
        int m = X.numCols();
        SimpleMatrix dW = A_error.mult(this.X.transpose()).divide(m);

        // adjust the bias
        double dbVal = A_error.elementSum() / m;
        SimpleMatrix db = new SimpleMatrix(b.numRows(), 1);
        for (int r=0;r<b.numRows();r++) {
            db.set(r, 0, dbVal);
        }

        return new Pair<>(dW, db);
    }

    /**
     * Update weights and biases
     *
     * @param dW - deltas for weights
     * @param db - deltas for biases
     */
    public void updateWeightsAndBias(SimpleMatrix dW, SimpleMatrix db) {
        w = w.minus(dW);
        b = b.minus(db);
    }
}

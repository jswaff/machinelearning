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

    private SimpleMatrix w;  // weights matrix, j x k where j = units this layer, k = prev. layer
    private SimpleMatrix b;  // bias column vector, j x 1

    // cached during forward pass
    private SimpleMatrix X;  // input from previous layer, n x m, where n = features and m = training examples
    private SimpleMatrix Z;  // the linear computation portion of the output, j x m
    private SimpleMatrix A;  // output of this layer -- g(Z), j x m

    // cached during backward pass
    private SimpleMatrix dCdZ;
    private SimpleMatrix dCdW;
    private SimpleMatrix dCdb;

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
                w.set(r, c, rand.nextDouble()-0.5);
            }
        }
        b = new SimpleMatrix(numUnits, 1);
        for (int r=0;r<numUnits;r++) {
            b.set(r, 0, 0.0);
        }
    }

    public SimpleMatrix getWeights() { return w; }

    public Double getWeight(int unit, int prevUnit) {
        return w.get(unit, prevUnit);
    }

    public void setWeight(int unit, int prevUnit, Double val) {
        w.set(unit, prevUnit, val);
    }

    public SimpleMatrix getBiases() { return b; }

    public Double getBias(int unit) {
        return b.get(unit, 0);
    }

    public void setBias(int unit, Double val) {
        b.set(unit, 0, val);
    }

    public SimpleMatrix getX() { return X; }

    public SimpleMatrix getZ() { return Z; }

    public SimpleMatrix getA() { return A; }

    public SimpleMatrix get_dCdZ() { return dCdZ; }

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
     * Calculate the gradient of the cost function w.r.t. the weights and biases
     *
     *  By the chain rule in calculus, dC/dW = dZ/dW * dA/dZ * dC/dA
     *  dZ/dW: how much the input to this layer changes as the weights change
     *  dA/dZ: how much the output of this layer changes as the input changes
     *  dC/dA: how much the total cost changes as the output of this layer changes
     *
     * Similarly, dC/db = dZ/db * dA/dZ * dC/dA
     * dz/db: This is just 1
     *
     * @param dCdA the derivative of the cost function with respect to the output (activation)
     *
     * @return dCdW - the partial derivative of the cost function with respect to weights
     *         dCdB - the partial derivative of the cost function with respect to biases
     */
    public Pair<SimpleMatrix, SimpleMatrix> calculateGradients(SimpleMatrix dCdA) {

        int m = X.numCols();

        // adjust the weights
        SimpleMatrix dAdZ = calculateZPrime();
        dCdZ = dCdA.elementMult(dAdZ);
        dCdW = dCdZ.mult(this.X.transpose()).divide(m);

        // adjust the biases
        dCdb = new SimpleMatrix(b.numRows(), 1);
        for (int r=0;r<b.numRows();r++) {
            double dbVal = 0.0;
            for (int c=0;c<dCdZ.numCols();c++) {
                dbVal += dCdZ.get(r, c);
            }
            dCdb.set(r, 0, dbVal / m);
        }

        return new Pair<>(dCdW, dCdb);
    }

    /**
     * Update weights and biases
     *
     * @param  learningRate - the learning rate
     */
    public void updateWeightsAndBias(double learningRate) {
        // no multiply operator
        double reciprocalLearningRate = 1.0 / learningRate;
        w = w.minus(dCdW.divide(reciprocalLearningRate));
        b = b.minus(dCdb.divide(reciprocalLearningRate));
    }

    private SimpleMatrix calculateZPrime() {
        SimpleMatrix Z_prime = new SimpleMatrix(Z.numRows(), Z.numCols());
        // unfortunately no broadcast operator
        for (int r=0;r<Z_prime.numRows();r++) {
            for (int c=0;c<Z_prime.numCols();c++) {
                Z_prime.set(r, c, activationFunction.derivativeFunc(Z.get(r, c)));
            }
        }
        return Z_prime;
    }

}

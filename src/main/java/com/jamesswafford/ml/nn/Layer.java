package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import com.jamesswafford.ml.nn.activation.ActivationFunctionFactory;
import com.jamesswafford.ml.nn.util.MatrixUtil;
import lombok.Data;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

@RequiredArgsConstructor
public class Layer {

    @Getter
    private final int numUnits;

    @Getter
    private final ActivationFunction activationFunction;

    private INDArray w;  // weights matrix, j x k where j = units this layer, k = prev. layer
    private INDArray b;  // bias column vector, j x 1

    // cached during forward pass
    private INDArray X;  // input from previous layer, n x m, where n = features and m = training examples
    @Getter
    private INDArray Z;  // the linear computation portion of the output, j x m
    @Getter
    private INDArray A;  // output of this layer -- g(Z), j x m

    // cached during backward pass
    @Getter
    private INDArray dCdZ;
    private INDArray dCdW;
    private INDArray dCdb;

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
        w = Nd4j.zeros(DataType.DOUBLE, numUnits, numUnitsPreviousLayer);
        // TODO: initialize with rand() and then .subi(0.5)
        for (int r=0;r<numUnits;r++) {
            for (int c=0;c<numUnitsPreviousLayer;c++) {
                w.putScalar(r, c, rand.nextDouble()-0.5);
            }
        }
        b = Nd4j.zeros(DataType.DOUBLE, numUnits, 1);
    }

    public SimpleMatrix getWeights() { return MatrixUtil.transform(w); }

    public Double getWeight(int unit, int prevUnit) { // TODO: unbox
        return w.getDouble(unit, prevUnit);
    }

    public void setWeight(int unit, int prevUnit, Double val) { // TODO: unbox
        w.putScalar(unit, prevUnit, val);
    }

    public SimpleMatrix getBiases() { return MatrixUtil.transform(b); }

    public Double getBias(int unit) { // TODO: unbox
        return b.getDouble(unit, 0);
    }

    public void setBias(int unit, Double val) {
        b.putScalar(unit, 0, val);
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
        this.X = MatrixUtil.transform(X);

        Z = w.mmul(this.X).add(b); // TODO: the add does a copy

        // TODO: is there a better way to map the activation function?  Look into Transform Op
        double[][] a_vals = new double[Z.rows()][Z.columns()];
        for (int r=0;r<Z.rows();r++) {
            for (int c=0;c<Z.columns();c++) {
                a_vals[r][c] = activationFunction.func(Z.getDouble(r, c));
            }
        }
        A = Nd4j.create(a_vals);

        return new Pair<>(MatrixUtil.transform(Z), MatrixUtil.transform(A));
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

        int m = X.columns();
        SimpleMatrix my_X = MatrixUtil.transform(X);

        // adjust the weights
        SimpleMatrix dAdZ = calculateZPrime();
        SimpleMatrix my_dCdZ = dCdA.elementMult(dAdZ);
        SimpleMatrix my_dCdW = my_dCdZ.mult(my_X.transpose()).divide(m);

        // adjust the biases
        SimpleMatrix my_dCdb = new SimpleMatrix(b.rows(), 1);
        for (int r=0;r<b.rows();r++) {
            // TODO: a faster way to add across the row?
            double dbVal = 0.0;
            for (int c=0;c<my_dCdZ.numCols();c++) {
                dbVal += my_dCdZ.get(r, c);
            }
            my_dCdb.set(r, 0, dbVal / m);
        }

        this.dCdZ = MatrixUtil.transform(my_dCdZ);
        this.dCdW = MatrixUtil.transform(my_dCdW);
        this.dCdb = MatrixUtil.transform(my_dCdb);

        return new Pair<>(my_dCdW, my_dCdb);
    }

    /**
     * Update weights and biases
     *
     * @param  learningRate - the learning rate
     */
    public void updateWeightsAndBias(double learningRate) {
        // no multiply operator
        double reciprocalLearningRate = 1.0 / learningRate;
        w.subi(dCdW.div(reciprocalLearningRate));  // TODO: this does a copy
        b.subi(dCdb.div(reciprocalLearningRate));  // TODO: this does a copy
    }

    private SimpleMatrix calculateZPrime() {
        SimpleMatrix Z_prime = new SimpleMatrix(Z.rows(), Z.columns());

        // unfortunately no broadcast operator
        // TODO: is there a faster way to map the activation function?  Look at Transform Ops
        for (int r=0;r<Z_prime.numRows();r++) {
            for (int c=0;c<Z_prime.numCols();c++) {
                Z_prime.set(r, c, activationFunction.derivativeFunc(Z.getDouble(r, c)));
            }
        }
        return Z_prime;
    }

    public LayerState getState() {
        return new LayerState(this);
    }

    public static Layer fromState(LayerState state) {
        Layer layer = new Layer(state.numUnits, ActivationFunctionFactory.create(state.activationFunction));
        layer.w = Nd4j.create(state.weights);
        layer.b = Nd4j.create(state.biases);
        return layer;
    }

    @Data
    public static class LayerState {
        private int numUnits;
        private int prevUnits;
        private String activationFunction;
        private double[][] weights;
        private double[] biases;

        public LayerState(Layer layer) {
            this.numUnits = layer.numUnits;
            this.prevUnits = layer.w.columns();
            this.activationFunction = layer.activationFunction.getName();
            this.weights = layer.w.toDoubleMatrix();
            this.biases = layer.b.toDoubleVector();
        }
    }
}

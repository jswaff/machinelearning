package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import com.jamesswafford.ml.nn.activation.ActivationFunctionFactory;
import lombok.Data;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.javatuples.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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
        Nd4j.getRandom().setSeed(seed);
        w = Nd4j.rand(DataType.DOUBLE, numUnits, numUnitsPreviousLayer).subi(0.5);
        b = Nd4j.zeros(DataType.DOUBLE, numUnits, 1);
    }

    public INDArray getWeights() { return w; }

    public double getWeight(int unit, int prevUnit) {
        return w.getDouble(unit, prevUnit);
    }

    public void setWeight(int unit, int prevUnit, double val) {
        w.putScalar(unit, prevUnit, val);
    }

    public INDArray getBiases() { return b; }

    public double getBias(int unit) {
        return b.getDouble(unit, 0);
    }

    public void setBias(int unit, double val) {
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
    public Pair<INDArray, INDArray> feedForward(INDArray X) {
        this.X = X;

        Z = w.mmul(X).add(b);
        A = activationFunction.func(Z, true);

        return new Pair<>(Z, A);
    }

    /**
     * Calculate the gradient of the cost function w.r.t. the weights and biases
     * By the chain rule in calculus, dC/dW = dZ/dW * dA/dZ * dC/dA
     * dZ/dW: how much the input to this layer changes as the weights change
     * dA/dZ: how much the output of this layer changes as the input changes
     * dC/dA: how much the total cost changes as the output of this layer changes
     * Similarly, dC/db = dZ/db * dA/dZ * dC/dA
     * dz/db: This is just 1
     *
     * @param dCdA the derivative of the cost function with respect to the output (activation)
     *
     * @return dCdW - the partial derivative of the cost function with respect to weights
     *         dCdB - the partial derivative of the cost function with respect to biases
     */
    public Pair<INDArray, INDArray> calculateGradients(INDArray dCdA) {

        int m = X.columns();

        // adjust the weights
        INDArray dAdZ = activationFunction.derivativeFunc(Z, true);
        dCdZ = dCdA.mul(dAdZ);
        dCdW = dCdZ.mmul(X.transpose()).div(m);

        // adjust the biases
        dCdb = dCdZ.sum(1).reshape(b.rows(),1).div(m);

        return new Pair<>(dCdW, dCdb);
    }

    /**
     * Update weights and biases
     *
     * @param  learningRate - the learning rate
     */
    public void updateWeightsAndBias(double learningRate) {
        w.subi(dCdW.mul(learningRate));
        b.subi(dCdb.mul(learningRate));
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
        private double[][] biases;

        public LayerState(Layer layer) {
            this.numUnits = layer.numUnits;
            this.prevUnits = layer.w.columns();
            this.activationFunction = layer.activationFunction.getName();
            this.weights = layer.w.toDoubleMatrix();
            this.biases = layer.b.toDoubleMatrix();
        }
    }
}

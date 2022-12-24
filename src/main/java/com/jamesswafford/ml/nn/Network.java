package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.cost.CostFunction;
import lombok.Builder;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.*;

@RequiredArgsConstructor
@Builder
public class Network {

    @NonNull
    private final int numInputUnits;

    @NonNull
    private final List<Layer> layers;

    @NonNull
    private final CostFunction costFunction;

    /**
     * Initialize the network
     */
    public void initialize() {
        int numUnitsPrevLayer = numInputUnits;
        for (Layer layer : layers) {
            layer.initialize(numUnitsPrevLayer);
            numUnitsPrevLayer = layer.getNumUnits();
        }
    }

    /**
     * Train the network
     * Note- the network should already be initialized.
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @param Y - labels, of shape 1 x m
     * @param numEpochs - the number of epochs
     */
    public void train(SimpleMatrix X, SimpleMatrix Y, int numEpochs) {

        List<Layer> reverseLayers = new ArrayList<>(layers);
        Collections.reverse(reverseLayers);

        for (int i=0;i<numEpochs;i++) {

            // feed forward
            SimpleMatrix A = X;
            for (Layer layer : layers) {
                Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.feedForward(A);
                A = Z_A.getValue1();
            }

            // backwards propagation

            // we want to find the derivative of the cost function with respect to each weight and bias
            // by the chain rule in calculus, dC/dW = dZ/dW * dA/dZ * dC/dA
            // dZ/dW: how much the input to a neuron changes as the weight changes
            //        this is the output of the previous layer
            // dA/dZ: how much the output to the neuron changes as the input changes
            //        this is the derivative of the activation function
            // dC/dA: how much the cost changes as the output to the neuron changes
            //        for the last layer this is simply the derivative of the cost function
            //        for previous layers it's more complex.  the change in weight will impact the output of
            //        all neurons in the next layer, so the change to the cost function is the sum
            //        of the change with respect to each neuron individually.

            SimpleMatrix dCdA = A.minus(Y).divide(0.5); // derivative of quadratic cost function is 2(A-Y)

            for (int L=0;L< reverseLayers.size();L++) {
                Layer layer = reverseLayers.get(L);
                Pair<SimpleMatrix, SimpleMatrix> dCdW_dCdB = layer.calculateUpdatedWeightsAndBiases(dCdA);
                SimpleMatrix dCdW = dCdW_dCdB.getValue0();
                SimpleMatrix dCdb = dCdW_dCdB.getValue1();

                // update the dCdA component of the backprop calculation for the previous layer (l-1)
                if (L < reverseLayers.size()-1) {
                    SimpleMatrix dAdZ = layer.calculateZPrime();
                    SimpleMatrix dCdZ = dCdA.elementMult(dAdZ);
                    dCdA = layer.getWeights().transpose().mult(dCdZ);
                }

                // update weights and biases
                // TODO: consider doing this after the entire back prop operation is complete
                // particularly when doing mini-batches: should they all process before doing any updates?
                layer.updateWeightsAndBias(dCdW, dCdb);
            }

            System.out.println("\tcost(" + i + "): " + cost(predict(X), Y));
        }
    }

    /**
     * Predict the output
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @return prediction matrix, of shape 1 x m
     */
    public SimpleMatrix predict(SimpleMatrix X) {

        SimpleMatrix A = X;
        for (Layer layer : layers) {
            A = layer.feedForward(A).getValue1();
        }

        return A;
    }

    public double cost(SimpleMatrix predictions, SimpleMatrix labels) {
        return costFunction.cost(predictions, labels);
    }
}

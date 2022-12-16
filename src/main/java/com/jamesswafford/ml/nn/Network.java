package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.cost.CostFunction;
import lombok.Builder;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
                A = layer.activationForward(A);
                // TODO: cache Z,A per layer
            }

            // TODO: calculate cost (use test data)
//        double cost = costFunction.averageCost(A, Y);
//        System.out.println("cost: " + cost);

            // backwards propagation
            // dZ = A - Y

            for (Layer layer : reverseLayers) {
                // get cached Z, A for this layer
                // dZ, dW, db = layer.backprop(dZ, Z, A)
                // layer.updateWeightsAndBiases(dW, db)
            }
        }
    }

    /**
     * Predict the output
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @return prediction matrix, of shape 1 x m
     */
    public SimpleMatrix predict(SimpleMatrix X) {

        return new SimpleMatrix(1, X.numCols());
    }

}

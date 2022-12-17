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
            SimpleMatrix dA = A.minus(Y);

            for (int L=0;L< reverseLayers.size();L++) {
                Layer layer = reverseLayers.get(L);
                Pair<SimpleMatrix, SimpleMatrix> dW_db = layer.backProp(dA);
                if (L < reverseLayers.size()-1) {
                    // calculate dA for previous layer
                    Layer previousLayer = reverseLayers.get(L+1);
                    SimpleMatrix Z_prime = previousLayer.getZPrime();
                    dA = layer.getWeights().transpose().mult(dA).elementMult(Z_prime);
                }
                layer.updateWeightsAndBias(dW_db.getValue0(), dW_db.getValue1());
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

        SimpleMatrix A = X;
        for (Layer layer : layers) {
            A = layer.feedForward(A).getValue1();
        }

        return A;
    }

}

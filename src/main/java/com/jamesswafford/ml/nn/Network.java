package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.cost.CostFunction;
import lombok.Builder;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Triplet;

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

        Map<Layer, SimpleMatrix> Z_cache = new HashMap<>();
        Map<Layer, SimpleMatrix> A_cache = new HashMap<>();
        for (int i=0;i<numEpochs;i++) {
            // feed forward
            SimpleMatrix A = X;
            for (Layer layer : layers) {
                Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.activationForward(A);
                A = Z_A.getValue1();
                // cache the results for use in back prop
                Z_cache.put(layer, Z_A.getValue0());
                A_cache.put(layer, A);
            }

            // TODO: calculate cost (use test data)
//        double cost = costFunction.averageCost(A, Y);
//        System.out.println("cost: " + cost);

            // backwards propagation
            SimpleMatrix dA = A.minus(Y);

            for (Layer layer : reverseLayers) {
                SimpleMatrix Z = Z_cache.get(layer);
                Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> dA_dW_db = layer.backProp(dA, Z);
                dA = dA_dW_db.getValue0();
                layer.updateWeightsAndBias(dA_dW_db.getValue1(), dA_dW_db.getValue2());
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
            A = layer.activationForward(A).getValue1();
        }

        return A;
    }

}

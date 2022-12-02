package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.cost.CostFunction;
import lombok.Builder;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;

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

    public void initialize() {
        int numUnitsPrevLayer = numInputUnits;
        for (Layer layer : layers) {
            layer.initialize(numUnitsPrevLayer);
            numUnitsPrevLayer = layer.getNumUnits();
        }
    }

    /**
     * Train the network on inputs X and labels Y.  The shape of X should be m,n, where m is the number of training
     * examples and n is the number of features.  The shape of Y should be m x 1.
     */
    public void train(SimpleMatrix X, SimpleMatrix Y) {
        // TODO: - number of epochs, minibatches
        SimpleMatrix prevA = X;
        SimpleMatrix A_out;
        for (Layer layer : layers) {
            A_out = layer.activationForward(prevA);
            prevA = A_out;
        }

        // TODO - cost

        // TODO - backprop
    }


    // TODO - cost (AL, Y)
    // AL - activation for layer L, shape (1, number of examples)
    // Y - shape (1, number of examples)
    // returns -- cost


}

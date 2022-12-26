package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.cost.CostFunction;
import com.jamesswafford.ml.nn.util.DataSplitter;
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
     * @param X_train - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @param Y_train - labels, of shape L x m, where L is the number of outputs and m is the number of training examples
     * @param numEpochs - the number of epochs
     * @param miniBatchSize - the size of the mini batches.  Note the last batch may be smaller.
     * @param learningRate - the learning rate.
     * @param X_test - test samples (optional).  If provided, the cost will be output every 10 epochs.
     * @param Y_test - test labels (optional)
     */
    public void train(SimpleMatrix X_train, SimpleMatrix Y_train, int numEpochs, int miniBatchSize, double learningRate,
                      SimpleMatrix X_test, SimpleMatrix Y_test)
    {
        int m = X_train.numCols(); // number of training samples

        List<Layer> reverseLayers = new ArrayList<>(layers);
        Collections.reverse(reverseLayers);

        // split the data up into mini-batches
        int numMiniBatches = m / miniBatchSize;
        if ((X_train.numCols() % m) != 0) {
            numMiniBatches++;
        }

        for (int i=0;i<numEpochs;i++) {

            for (int j=0;j<numMiniBatches;j++) {

                Pair<SimpleMatrix, SimpleMatrix> X_Y_batch = DataSplitter.getMiniBatch(X_train, Y_train, j, miniBatchSize);
                SimpleMatrix X_batch = X_Y_batch.getValue0();
                SimpleMatrix Y_batch = X_Y_batch.getValue1();

                // feed forward
                SimpleMatrix A = X_batch;
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

                SimpleMatrix dCdA = A.minus(Y_batch).divide(0.5); // derivative of quadratic cost function is 2(A-Y)

                for (int L = 0; L < reverseLayers.size(); L++) {
                    Layer layer = reverseLayers.get(L);
                    Pair<SimpleMatrix, SimpleMatrix> dCdW_dCdB = layer.calculateUpdatedWeightsAndBiases(dCdA);
                    SimpleMatrix dCdW = dCdW_dCdB.getValue0();
                    SimpleMatrix dCdb = dCdW_dCdB.getValue1();

                    // update the dCdA component of the backprop calculation for the previous layer (l-1)
                    if (L < reverseLayers.size() - 1) {
                        SimpleMatrix dAdZ = layer.calculateZPrime();
                        SimpleMatrix dCdZ = dCdA.elementMult(dAdZ);
                        dCdA = layer.getWeights().transpose().mult(dCdZ);
                    }

                    // update weights and biases
                    double normalizedLearningRate = learningRate / X_batch.numCols();
                    layer.updateWeightsAndBias(dCdW, dCdb, normalizedLearningRate);
                }
            }

            if (X_test != null && Y_test != null && (i % 10) == 0) {
                System.out.println("\tcost(" + i + "): " + cost(predict(X_test), Y_test));
            }
        }
    }

    /**
     * Predict the output
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @return prediction matrix, of shape L x m, where L is the number of outputs and m is the number of training examples
     */
    public SimpleMatrix predict(SimpleMatrix X) {

        SimpleMatrix A = X;
        for (Layer layer : layers) {
            A = layer.feedForward(A).getValue1();
        }

        return A;
    }

    /**
     * Calculate the cost (error) of the predictions vs the "ground truth" labels.
     *
     * @param predictions - network output
     * @param labels - ground truth
     *
     * @return - the cost
     */
    public double cost(SimpleMatrix predictions, SimpleMatrix labels) {
        return costFunction.cost(predictions, labels);
    }
}

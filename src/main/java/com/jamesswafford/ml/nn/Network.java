package com.jamesswafford.ml.nn;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.jamesswafford.ml.nn.cost.CostFunction;
import com.jamesswafford.ml.nn.cost.CostFunctionFactory;
import com.jamesswafford.ml.nn.util.StopEvaluator;
import lombok.*;
import org.javatuples.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.jamesswafford.ml.nn.util.DataSplitter.getMiniBatch;

@Getter
@RequiredArgsConstructor
@Builder
public class Network {

    private final int numInputUnits;

    private final List<Layer> layers;

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
     * This method requires the training and test sets be loaded into memory, which is not very memory efficient
     * for large data sets.  For a more memory efficient alternative, see the other train() function which
     * uses a callback to retrieve one mini-batch at a time.
     *
     * @param X_train - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @param Y_train - labels, of shape L x m, where L is the number of outputs and m is the number of training examples
     * @param numEpochs - the number of epochs
     * @param miniBatchSize - the size of the mini batches.  Note the last batch may be smaller
     * @param learningRate - the learning rate
     * @param X_test - test samples (optional).  If provided, the cost will be output every 10 epochs
     * @param Y_test - test labels (optional)
     *
     * @return - the final network state
     */
    public NetworkState train(double[][] X_train, double[][] Y_train, int numEpochs, int miniBatchSize, double learningRate,
                              double[][] X_test, double[][] Y_test)
    {
        return train(Nd4j.create(X_train), Nd4j.create(Y_train), numEpochs, miniBatchSize, learningRate,
                Nd4j.create(X_test), Nd4j.create(Y_test));
    }
    public NetworkState train(INDArray X_train, INDArray Y_train, int numEpochs, int miniBatchSize, double learningRate,
                      INDArray X_test, INDArray Y_test)
    {
        int m = X_train.columns(); // number of training samples

        // split the data up into mini-batches
        int numMiniBatches = m / miniBatchSize;
        if ((X_train.columns() % miniBatchSize) != 0) {
            numMiniBatches++;
        }

        return train(numMiniBatches, batchNum -> getMiniBatch(X_train, Y_train, batchNum, miniBatchSize),
                numEpochs, learningRate, X_test, Y_test);
    }

    /**
     * Train the network
     * Note- the network should already be initialized.
     *
     * @param numMiniBatches - the number of mini batches.  Note the last batch may be smaller
     * @param miniBatchFunc - callback function to retrieve one mini-batch of training data
     * @param numEpochs - the number of epochs
     * @param learningRate - the learning rate
     * @param X_test - test samples (optional).  If provided, the cost will be output every 10 epochs
     * @param Y_test - test labels (optional)
     *
     * @return - the final network state
     */
    public NetworkState train(int numMiniBatches, Function<Integer, Pair<double[][], double[][]>> miniBatchFunc,
                              int numEpochs, double learningRate, double[][] X_test, double[][] Y_test)
    {
        Function<Integer, Pair<INDArray, INDArray>> mbf = (X) -> {
            Pair<double[][],double[][]> p = miniBatchFunc.apply(X);
            return new Pair<>(Nd4j.create(p.getValue0()), Nd4j.create(p.getValue1()));
        };
        return train(numMiniBatches, mbf, numEpochs, learningRate, Nd4j.create(X_test), Nd4j.create(Y_test));
    }

    public NetworkState train(int numMiniBatches, Function<Integer, Pair<INDArray, INDArray>> miniBatchFunc,
                      int numEpochs, double learningRate, INDArray X_test, INDArray Y_test)
    {
        StopEvaluator stopEvaluator = new StopEvaluator(this, 10, 1E-4);

        for (int i=0;i<numEpochs;i++) {

            // train the network
            for (int j=0;j<numMiniBatches;j++) {
                Pair<INDArray, INDArray> X_Y_batch = miniBatchFunc.apply(j);
                INDArray X_batch = X_Y_batch.getValue0();
                INDArray Y_batch = X_Y_batch.getValue1();
                processMiniBatch(X_batch, Y_batch, learningRate);
            }

            // if we have test data, calculate the cost and evaluate if we should stop
            if (X_test != null && Y_test != null && (i % 10) == 0) {

                // calculate the cost using the training data
                double costTraining = 0.0;
                for (int j=0;j<numMiniBatches;j++) {
                    Pair<INDArray, INDArray> X_Y_batch = miniBatchFunc.apply(j);
                    INDArray X_batch = X_Y_batch.getValue0();
                    INDArray Y_batch = X_Y_batch.getValue1();
                    costTraining += cost(predict(X_batch), Y_batch);
                }
                costTraining /= numMiniBatches;

                // calculate the cost using the test data
                double costTest = cost(predict(X_test), Y_test);

                System.out.println(i + "," + costTraining + "," + costTest);

                if (stopEvaluator.stop(costTest)) {
                    System.out.println("\tearly stop triggered");
                    return stopEvaluator.getBestNetwork();
                }
            }
        }

        return getState();
    }

    /**
     * Predict the correct labels
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @return prediction matrix, of shape L x m, where L is the number of outputs and m is the number of training examples
     */

    public double[][] predict(double[][] X) {
        return predict(Nd4j.create(X)).toDoubleMatrix();
    }
    public INDArray predict(INDArray X) {

        INDArray A = X;
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
    public double cost(double[][] predictions, double[][] labels) {
        return cost(Nd4j.create(predictions), Nd4j.create(labels));
    }
    public double cost(INDArray predictions, INDArray labels) {
        return costFunction.cost(predictions, labels);
    }

    public NetworkState getState() {
        return new NetworkState(this);
    }

    public static Network fromState(NetworkState state) {
        return Network.builder()
                .numInputUnits(state.numInputUnits)
                .costFunction(CostFunctionFactory.create(state.costFunction))
                .layers(Arrays.stream(state.layers).map(Layer::fromState).collect(Collectors.toList()))
                .build();
    }

    public String toJson() {
        GsonBuilder gsonBuilder = new GsonBuilder();
        gsonBuilder.setPrettyPrinting();
        return gsonBuilder.create().toJson(getState());
    }

    public static Network fromJson(String json) {
        Network.NetworkState state = new Gson().fromJson(json, Network.NetworkState.class);
        return fromState(state);
    }

    private void processMiniBatch(INDArray X_batch, INDArray Y_batch, double learningRate) {

        // feed forward
        INDArray A = X_batch;
        for (Layer layer : layers) {
            Pair<INDArray, INDArray> Z_A = layer.feedForward(A);
            A = Z_A.getValue1();
        }

        // backwards propagation
        // we update dC/dA as we go.  For the last layer, this is simply the derivative of the cost
        // function.  It's more complex for hidden layers, as changes to the activation function will
        // impact the output of each neuron in the next layer.
        double normalizedLearningRate = learningRate / X_batch.columns();
        INDArray dCdA = A.sub(Y_batch);

        for (int L = layers.size()-1; L >= 0; L--) {
            Layer layer = layers.get(L);
            layer.calculateGradients(dCdA);

            // set dC/dA for the previous layer (l-1)
            if (L > 0) {
                INDArray dCdZ = layer.getDCdZ();
                dCdA = layer.getWeights().transpose().mmul(dCdZ);
            }
        }

        // update the weights and biases
        layers.forEach(layer -> layer.updateWeightsAndBias(normalizedLearningRate));
    }

    @Data
    public static class NetworkState {
        private int numInputUnits;
        private String costFunction;
        private Layer.LayerState[] layers;

        public NetworkState(Network network) {
            this.numInputUnits = network.numInputUnits;
            this.costFunction = network.costFunction.getName();
            this.layers = new Layer.LayerState[network.layers.size()];
            for (int i=0;i<network.layers.size();i++) {
                this.layers[i] = network.layers.get(i).getState();
            }
        }
    }
}

package com.jamesswafford.ml.nn;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.jamesswafford.ml.nn.cost.CostFunction;
import com.jamesswafford.ml.nn.cost.CostFunctionFactory;
import com.jamesswafford.ml.nn.util.StopEvaluator;
import lombok.*;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.jamesswafford.ml.nn.util.DataSplitter.getMiniBatch;

@RequiredArgsConstructor
@Builder
public class Network {

    @Getter
    @NonNull
    private final int numInputUnits;

    @Getter
    @NonNull
    private final List<Layer> layers;

    @Getter
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
     */
    public void train(SimpleMatrix X_train, SimpleMatrix Y_train, int numEpochs, int miniBatchSize, double learningRate,
                      SimpleMatrix X_test, SimpleMatrix Y_test)
    {
        int m = X_train.numCols(); // number of training samples

        // split the data up into mini-batches
        int numMiniBatches = m / miniBatchSize;
        if ((X_train.numCols() % miniBatchSize) != 0) {
            numMiniBatches++;
        }

        train(numMiniBatches, batchNum -> getMiniBatch(X_train, Y_train, batchNum, miniBatchSize),
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
     */
    public void train(int numMiniBatches, Function<Integer, Pair<SimpleMatrix, SimpleMatrix>> miniBatchFunc,
                      int numEpochs, double learningRate, SimpleMatrix X_test, SimpleMatrix Y_test)
    {
        StopEvaluator stopEvaluator = new StopEvaluator(10, null);

        for (int i=0;i<numEpochs;i++) {

            for (int j=0;j<numMiniBatches;j++) {
                Pair<SimpleMatrix, SimpleMatrix> X_Y_batch = miniBatchFunc.apply(j);
                SimpleMatrix X_batch = X_Y_batch.getValue0();
                SimpleMatrix Y_batch = X_Y_batch.getValue1();

                processMinibatch(X_batch, Y_batch, learningRate);
            }

            if (X_test != null && Y_test != null && (i % 10) == 0) {
                double cost = cost(predict(X_test), Y_test);
                System.out.println("\tcost(" + i + "): " + cost);
                if (stopEvaluator.stop(cost)) {
                    System.out.println("\tearly stop triggered");
                    break;
                }
            }
        }
    }

    /**
     * Predict the correct labels
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

    private void processMinibatch(SimpleMatrix X_batch, SimpleMatrix Y_batch, double learningRate) {

        // feed forward
        SimpleMatrix A = X_batch;
        for (Layer layer : layers) {
            Pair<SimpleMatrix, SimpleMatrix> Z_A = layer.feedForward(A);
            A = Z_A.getValue1();
        }

        // backwards propagation
        // we update dC/dA as we go.  For the last layer, this is simply the derivative of the cost
        // function.  It's more complex for hidden layers, as changes to the activation function will
        // impact the output of each neuron in the next layer.
        double normalizedLearningRate = learningRate / X_batch.numCols();
        SimpleMatrix dCdA = A.minus(Y_batch);

        for (int L = layers.size()-1; L >= 0; L--) {
            Layer layer = layers.get(L);
            layer.calculateGradients(dCdA);

            // set dC/dA for the previous layer (l-1)
            if (L > 0) {
                SimpleMatrix dCdZ = layer.getDCdZ();
                dCdA = layer.getWeights().transpose().mult(dCdZ);
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

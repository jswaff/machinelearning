package com.jamesswafford.ml.nn.cost;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class MSE implements CostFunction {

    public static MSE INSTANCE = new MSE();

    private MSE() {
    }

    @Override
    public String getName() {
        return "mse";
    }

    /**
     * Calculate the cost (error)
     *
     * @param predictions l x m matrix, where l is the number of units in the output layer and m is the number
     *                    of predictions (one per network input)
     * @param labels  l x m matrix
     * @return the cost over all training examples.
     */
    @Override
    public double cost(INDArray predictions, INDArray labels) {
        if (!Arrays.equals(predictions.shape(), labels.shape())) {
            throw new IllegalStateException("Shapes do not match.  labels: " +
                    Arrays.toString(labels.shape()) + "; predictions: " +
                    Arrays.toString(predictions.shape()));
        }
        INDArray e = predictions.sub(labels);
        INDArray esq = Transforms.pow(e, 2, false);
        return esq.divi(2.0).sumNumber().doubleValue() / predictions.columns();
    }
}

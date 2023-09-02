package com.jamesswafford.ml.nn.util;

import org.javatuples.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class DataSplitter {

    /**
     * Retrieve a mini-batch from the X and Y matrices.
     *
     * @param X - input matrix of shape n x m, where n is the number of features and m is the number of training examples
     * @param Y - labels, of shape L x m, where L is the number of outputs and m is the number of training examples
     * @param batchNumber - which batch to retrieve.  Note this is 0 indexed.
     * @param batchSize - the number of samples to include in the batch.  Note the output may be smaller on the last batch.
     *
     * @return the X, Y matrices of for the mini-batch.  The X matrix wll have dimensions n x batch size, and the
     *    Y matrix will have dimensions 1 x batch size.
     */
    public static Pair<INDArray, INDArray> getMiniBatch(INDArray X, INDArray Y, int batchNumber, int batchSize) {

        int startInd = batchNumber * batchSize; // inclusive
        int endInd = (batchNumber+1) * batchSize; // exclusive
        if (endInd > X.columns()) {
            endInd = X.columns();
        }

        INDArray X_batch = X.get(NDArrayIndex.all(), NDArrayIndex.interval(startInd, endInd));
        INDArray Y_batch = Y.get(NDArrayIndex.all(), NDArrayIndex.interval(startInd, endInd));

        return new Pair<>(X_batch, Y_batch);
    }

}

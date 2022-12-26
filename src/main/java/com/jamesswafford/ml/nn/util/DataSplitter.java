package com.jamesswafford.ml.nn.util;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

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
    public static Pair<SimpleMatrix, SimpleMatrix> getMiniBatch(SimpleMatrix X, SimpleMatrix Y, int batchNumber,
                                                                int batchSize) {

        int startInd = batchNumber * batchSize; // inclusive
        int endInd = (batchNumber+1) * batchSize; // exclusive
        if (endInd > X.numCols()) {
            endInd = X.numCols();
        }
        int actualBatchSize = endInd - startInd;

        SimpleMatrix X_batch = new SimpleMatrix(X.numRows(), actualBatchSize);
        SimpleMatrix Y_batch = new SimpleMatrix(Y.numRows(), actualBatchSize);

        for (int r=0;r<X.numRows();r++) {
            int batchCol=0;
            for (int c=startInd;c<endInd;c++) {
                X_batch.set(r, batchCol++, X.get(r, c));
            }
        }

        for (int r=0;r<Y.numRows();r++) {
            int batchCol=0;
            for (int c=startInd;c<endInd;c++) {
                Y_batch.set(r, batchCol++, Y.get(r, c));
            }
        }

        return new Pair<>(X_batch, Y_batch);
    }

}

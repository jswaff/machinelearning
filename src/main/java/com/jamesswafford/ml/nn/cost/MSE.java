package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;

public class MSE implements CostFunction {

    /**
     * Calculate the cost (error)
     *
     * @param predictions l x m matrix, where l is the number of units in the output layer and m is the number
     *                    of predictions (one per network input)
     * @param labels  l x m matrix
     * @return the cost over all training examples.
     */
    @Override
    public Double cost(SimpleMatrix predictions, SimpleMatrix labels) {
        if (predictions.numRows() != labels.numRows() || predictions.numCols() != labels.numCols()) {
            throw new IllegalStateException("Shapes do not match.  labels: " + labels.numRows() + " x " +
                    labels.numCols() + "; predictions: " + predictions.numRows() + " x " + predictions.numCols());
        }

        // 1/2 constant to cancel out the exponent when differentiating
        SimpleMatrix e = predictions.minus(labels).elementPower(2).divide(2.0);

        return e.elementSum() / predictions.numCols();
    }
}

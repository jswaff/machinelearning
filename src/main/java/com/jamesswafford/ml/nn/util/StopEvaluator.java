package com.jamesswafford.ml.nn.util;

import lombok.Getter;

import java.util.LinkedList;

/**
 * Utility to determine if training should be stopped.
 * Borrowed from https://bitbucket.org/tobias_hill/mnist-example
 */
public class StopEvaluator {

    private final int windowSize;
    private final Double acceptableErrorRate;
    private final LinkedList<Double> errorRates;

    @Getter
    private double lowestErrorRate = Double.MAX_VALUE;

    private double lastErrorAverage = Double.MAX_VALUE;

    public StopEvaluator(int windowSize, Double acceptableErrorRate) {
        this.windowSize = windowSize;
        this.acceptableErrorRate = acceptableErrorRate;
        this.errorRates = new LinkedList<>();
    }

    public boolean stop(double errorRate) {
        if (errorRate < lowestErrorRate) {
            lowestErrorRate = errorRate;
            // TODO: save network
        }

        if (acceptableErrorRate != null && lowestErrorRate < acceptableErrorRate) {
            return true;
        }

        // update rolling average
        errorRates.addLast(errorRate);

        if (errorRates.size() < windowSize) {
            return false;
        }

        if (errorRates.size() > windowSize) {
            errorRates.removeFirst();
        }

        double avg = calculateAverage();

        if (avg > lastErrorAverage) {
            return true;
        } else {
            lastErrorAverage = avg;
            return false;
        }

    }

    private double calculateAverage() {
        return errorRates.stream().mapToDouble(Double::doubleValue).average().getAsDouble();
    }
}

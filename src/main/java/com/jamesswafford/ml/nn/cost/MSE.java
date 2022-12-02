package com.jamesswafford.ml.nn.cost;

import java.util.ArrayList;
import java.util.List;

public class MSE implements CostFunction {

    @Override
    public List<Double> cost(List<Double> predictions, List<Double> labels) {

        if (predictions.size() != labels.size()) {
            throw new IllegalArgumentException("predictions and labels must be the same size");
        }

        List<Double> cost = new ArrayList<>();
        for (int i=0;i<predictions.size();i++) {
            Double e = predictions.get(i) - labels.get(i);
            cost.add(e * e);
        }

        return cost;
    }

    @Override
    public Double totalCost(List<Double> predictions, List<Double> labels) {
        return cost(predictions, labels).stream().reduce(0.0, Double::sum, Double::sum) / predictions.size();
    }
}

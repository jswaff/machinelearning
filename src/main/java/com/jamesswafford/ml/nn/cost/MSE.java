package com.jamesswafford.ml.nn.cost;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
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

    @Override
    public Double totalCost(SimpleMatrix predictions, SimpleMatrix labels) {
        // TODO: predictions for multiple examples
        List<Double> ps = new ArrayList<>();
        List<Double> ls = new ArrayList<>();
        for (int r=0;r< predictions.numRows();r++) {
            ps.add(predictions.get(r, 0));
            ls.add(labels.get(r, 0));
        }
        return totalCost(ps, ls);
    }
}

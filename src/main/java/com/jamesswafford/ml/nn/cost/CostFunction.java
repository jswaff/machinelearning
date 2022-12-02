package com.jamesswafford.ml.nn.cost;

import java.util.List;

public interface CostFunction {

    List<Double> cost(List<Double> predictions, List<Double> labels);

    Double totalCost(List<Double> predictions, List<Double> labels);

}

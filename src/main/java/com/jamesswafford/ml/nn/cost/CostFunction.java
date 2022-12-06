package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;

import java.util.List;

public interface CostFunction {

    List<Double> cost(List<Double> predictions, List<Double> labels);

    Double totalCost(List<Double> predictions, List<Double> labels);

    Double totalCost(SimpleMatrix predictions, SimpleMatrix labels);

}

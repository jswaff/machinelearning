package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface CostFunction {

    String getName();

    double cost(SimpleMatrix predictions, SimpleMatrix labels);

    double cost(INDArray predictions, INDArray labels);
}

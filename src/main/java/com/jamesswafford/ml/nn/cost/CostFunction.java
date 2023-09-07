package com.jamesswafford.ml.nn.cost;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface CostFunction {

    String getName();

    double cost(INDArray predictions, INDArray labels);
}

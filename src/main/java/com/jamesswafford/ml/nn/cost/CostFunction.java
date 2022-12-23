package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;

public interface CostFunction {

    Double cost(SimpleMatrix predictions, SimpleMatrix labels);

}

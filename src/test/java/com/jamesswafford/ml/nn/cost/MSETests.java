package com.jamesswafford.ml.nn.cost;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class MSETests {

    private final MSE mse = new MSE();

    @Test
    public void cost() {

        assertEquals(mse.cost(Collections.emptyList(), Collections.emptyList()), Collections.emptyList());

        List<Double> predictions = Arrays.asList(0.8, 0.4, 0.1);
        List<Double> labels = Arrays.asList(1.0, 1.0, 0.0);

        assertDoubleEquals(Arrays.asList(0.0, 0.0, 0.0), mse.cost(predictions, predictions));
        assertDoubleEquals(Arrays.asList(0.04, 0.36, 0.01), mse.cost(predictions, labels));
    }

    @Test
    public void totalCost() {
        List<Double> predictions = Arrays.asList(0.8, 0.4, 0.1);
        List<Double> labels = Arrays.asList(1.0, 1.0, 0.0);

        assertDoubleEquals(0.0, mse.totalCost(predictions, predictions));
        assertDoubleEquals((0.04+0.36+0.01)/3.0, mse.totalCost(predictions, labels) );
    }

}

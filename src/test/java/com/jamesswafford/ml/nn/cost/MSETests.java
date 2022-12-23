package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class MSETests {

    private final MSE mse = new MSE();

    @Test
    public void cost() {

        // labels for 3 outputs and 4 examples
        SimpleMatrix Y = new SimpleMatrix(3, 4, true,
                new double[]{ 0, 1, 0, 1,
                              0, 1, 0, 1,
                              0, 1, 0, 1});


        SimpleMatrix P = new SimpleMatrix(3, 4, true,
                new double[]{ 0, 1, 0, 1,
                              0, 1, 0, 1,
                              0, 1, 0, 1});

        double cost = mse.cost(P, Y);
        assertDoubleEquals(0.0, cost);

        // inverted - each of the 4 examples is completely wrong and gets a score of 3.0 each
        SimpleMatrix P2 = new SimpleMatrix(3, 4, true,
                new double[]{ 1, 0, 1, 0,
                              1, 0, 1, 0,
                              1, 0, 1, 0});

        double cost2 = mse.cost(P2, Y);
        assertDoubleEquals(3.0, cost2);

        // just one output is 50% wrong.  the other two are correct.
        // the sum of errors for that column is 0.25 * 3 = 0.75.
        // 0.75 / 4 = 0.1875
        SimpleMatrix P3 = new SimpleMatrix(3, 4, true,
                new double[]{ 0, 0.5, 0, 1,
                              0, 0.5, 0, 1,
                              0, 0.5, 0, 1});

        double cost3 = mse.cost(P3, Y);
        assertDoubleEquals(0.1875, cost3);

        // two outputs are 50% wrong.  the other two are correct.
        // the sum of errors is now 1.5.
        // 1.5 / 4 = 0.375
        SimpleMatrix P4 = new SimpleMatrix(3, 4, true,
                new double[]{ 0.5, 0.5, 0, 1,
                              0.5, 0.5, 0, 1,
                              0.5, 0.5, 0, 1});

        double cost4 = mse.cost(P4, Y);
        assertDoubleEquals(0.375, cost4);
    }
}

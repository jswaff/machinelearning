package com.jamesswafford.ml.nn.cost;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class MSETests {
    private static final double epsilon = 0.00001;
    private final MSE mse = MSE.INSTANCE;

    @Test
    public void cost_nd4j() {
        INDArray Y = Nd4j.create(new double[]{
                0, 1, 0, 1,
                0, 1, 0, 1,
                0, 1, 0, 1}, new int[]{3, 4});
        INDArray P = Nd4j.create(new double[]{
                    0, 1, 0, 1,
                    0, 1, 0, 1,
                    0, 1, 0, 1}, new int[]{3, 4});

        double cost = mse.cost(Y, P);
        assertEquals(0.0, cost, epsilon);

        // inverted - each of the 4 examples is completely wrong and gets a score of 3.0 each
        INDArray P2 = Nd4j.create(new double[]{
                1, 0, 1, 0,
                1, 0, 1, 0,
                1, 0, 1, 0}, new int[]{3, 4});
        double cost2 = mse.cost(P2, Y);
        assertEquals(3.0/2, cost2, epsilon);

        // just one output is 50% wrong.  the other two are correct.
        // the sum of errors for that column is 0.25 * 3 = 0.75.
        // 0.75 / 4 = 0.1875
        INDArray P3 = Nd4j.create(new double[]{
                0, 0.5, 0, 1,
                0, 0.5, 0, 1,
                0, 0.5, 0, 1}, new int[]{3, 4});
        double cost3 = mse.cost(P3, Y);
        assertEquals(0.1875/2, cost3, epsilon);

        // two outputs are 50% wrong.  the other two are correct.
        // the sum of errors is now 1.5.
        // 1.5 / 4 = 0.375
        INDArray P4 = Nd4j.create(new double[]{
                0.5, 0.5, 0, 1,
                0.5, 0.5, 0, 1,
                0.5, 0.5, 0, 1}, new int[]{3, 4});
        double cost4 = mse.cost(P4, Y);
        assertEquals(0.375/2, cost4, epsilon);
    }

    @Test
    public void cost_ejml() {

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
        assertEquals(0.0, cost, epsilon);

        // inverted - each of the 4 examples is completely wrong and gets a score of 3.0 each
        SimpleMatrix P2 = new SimpleMatrix(3, 4, true,
                new double[]{ 1, 0, 1, 0,
                              1, 0, 1, 0,
                              1, 0, 1, 0});

        double cost2 = mse.cost(P2, Y);
        assertEquals(3.0/2, cost2, epsilon);

        // just one output is 50% wrong.  the other two are correct.
        // the sum of errors for that column is 0.25 * 3 = 0.75.
        // 0.75 / 4 = 0.1875
        SimpleMatrix P3 = new SimpleMatrix(3, 4, true,
                new double[]{ 0, 0.5, 0, 1,
                              0, 0.5, 0, 1,
                              0, 0.5, 0, 1});

        double cost3 = mse.cost(P3, Y);
        assertEquals(0.1875/2, cost3, epsilon);

        // two outputs are 50% wrong.  the other two are correct.
        // the sum of errors is now 1.5.
        // 1.5 / 4 = 0.375
        SimpleMatrix P4 = new SimpleMatrix(3, 4, true,
                new double[]{ 0.5, 0.5, 0, 1,
                              0.5, 0.5, 0, 1,
                              0.5, 0.5, 0, 1});

        double cost4 = mse.cost(P4, Y);
        assertEquals(0.375/2, cost4, epsilon);
    }
}

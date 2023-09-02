package com.jamesswafford.ml.nn.util;

import org.javatuples.Pair;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static com.jamesswafford.ml.nn.util.DataSplitter.getMiniBatch;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class DataSplitterTests {

    @Test
    public void splitTest1() {
        INDArray X = getX();
        INDArray Y = getY();

        Pair<INDArray, INDArray> X_Y_split = getMiniBatch(X, Y, 0, 10);
        INDArray X_split = X_Y_split.getValue0();
        assertArrayEquals(new double[][]{
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}},
                X_split.toDoubleMatrix());

        INDArray Y_split = X_Y_split.getValue1();
        assertArrayEquals(new double[]{  100, 101, 102, 103, 104, 105, 106, 107, 108, 109 },
                Y_split.toDoubleVector());
    }

    @Test
    public void splitTest2() {
        INDArray X = getX();
        INDArray Y = getY();

        Pair<INDArray, INDArray> X_Y_split = getMiniBatch(X, Y, 0, 5);
        INDArray X_split = X_Y_split.getValue0();
        assertArrayEquals(new double[][]{
                {0,  1,  2,  3,  4},
                {10, 11, 12, 13, 14},
                {20, 21, 22, 23, 24 }},
                X_split.toDoubleMatrix());

        INDArray Y_split = X_Y_split.getValue1();
        assertArrayEquals(new double[]{  100, 101, 102, 103, 104 },
                Y_split.toDoubleVector());
    }

    @Test
    public void splitTest3() {
        INDArray X = getX();
        INDArray Y = getY();

        Pair<INDArray, INDArray> X_Y_split = getMiniBatch(X, Y, 2, 3);
        INDArray X_split = X_Y_split.getValue0();
        assertArrayEquals(new double[][]{ {6, 7, 8}, {16, 17, 18}, {26, 27, 28 }},
                X_split.toDoubleMatrix());

        INDArray Y_split = X_Y_split.getValue1();
        assertArrayEquals(new double[]{  106, 107, 108 },
                Y_split.toDoubleVector());
    }

    @Test
    public void splitTest4() {
        INDArray X = getX();
        INDArray Y = getY();

        Pair<INDArray, INDArray> X_Y_split = getMiniBatch(X, Y, 3, 3);
        INDArray X_split = X_Y_split.getValue0();
        assertArrayEquals(new double[][]{{9}, {19}, {29}},
                X_split.toDoubleMatrix());

        INDArray Y_split = X_Y_split.getValue1();
        assertArrayEquals(new double[]{  109 },
                Y_split.toDoubleVector());
    }

    private static INDArray getX() {
        return Nd4j.create(new double[][] {
                {0,  1,  2,  3,  4,  5,  6,  7,  8,  9},
                {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
                {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
        });
    }

    private static INDArray getY() {
        return Nd4j.create(new double[] { 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 });
    }
}

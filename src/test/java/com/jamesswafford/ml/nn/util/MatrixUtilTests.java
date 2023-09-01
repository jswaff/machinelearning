package com.jamesswafford.ml.nn.util;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MatrixUtilTests {

    @Test
    public void transformVectorTest() {
        SimpleMatrix Y = getY();
        INDArray Y2 = MatrixUtil.transform(Y);
        SimpleMatrix Y3 = MatrixUtil.transform(Y2);
        Assertions.assertArrayEquals(Y.getDDRM().getData(), Y3.getDDRM().getData());
    }

    @Test
    public void transformMatrixTest() {
        SimpleMatrix X = getX();
        INDArray X2 = MatrixUtil.transform(X);
        SimpleMatrix X3 = MatrixUtil.transform(X2);
        Assertions.assertArrayEquals(X.getDDRM().getData(), X3.getDDRM().getData());
    }

    @Test
    public void flattenTest() {
        double[][] arr = new double[][] {
                { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9}, { 10, 11, 12 }
        };
        double[] f = MatrixUtil.flatten(arr, 4, 3);
        Assertions.assertEquals(1, f[0]);
        Assertions.assertArrayEquals(new double[]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, f);
    }

    private static SimpleMatrix getX() {
        return new SimpleMatrix(3, 10, true,
                new double[]{
                        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29
                });
    }

    private static SimpleMatrix getY() {
        return new SimpleMatrix(1, 10, true,
                new double[] { 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 });
    }

}

package com.jamesswafford.ml.nn.util;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class MatrixUtilTests {

    @Test
    public void flattenTest() {
        double[][] arr = new double[][] {
                { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9}, { 10, 11, 12 }
        };
        double[] f = MatrixUtil.flatten(arr, 4, 3);
        Assertions.assertEquals(1, f[0]);
        Assertions.assertArrayEquals(new double[]{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, f);
    }


}

package com.jamesswafford.ml.nn.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class MatrixUtil {

    public static double[] flatten(double[][] matrix, int rows, int cols) {
        Double[][] m = new Double[rows][cols];
        for (int r=0;r<rows;r++) {
            for (int c=0;c<cols;c++) {
                m[r][c] = matrix[r][c];
            }
        }
        Double[] d = flattenStream(m).toArray(Double[] ::new);
        double[] d2 = new double[d.length];
        for (int i=0;i<d.length;i++) {
            d2[i] = d[i];
        }
        return d2;
    }

    private static <T> Stream<T> flattenStream(T[][] arrays) {
        List<T> list = new ArrayList<>();
        for (T[] array : arrays) {
            list.addAll(Arrays.asList(array));
        }

        return list.stream();
    }
}

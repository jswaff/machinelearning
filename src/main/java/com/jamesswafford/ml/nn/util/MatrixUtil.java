package com.jamesswafford.ml.nn.util;

import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class MatrixUtil {

    public static INDArray transform(SimpleMatrix matrix) {
        return Nd4j.create(matrix.getDDRM().data,
                new int[] { matrix.getDDRM().numRows, matrix.getDDRM().numCols });
    }

    public static SimpleMatrix transform(INDArray matrix) {
        double[] d;
        if (matrix.isVector()) {
            d = matrix.toDoubleVector();
        } else {
            double[][] m = matrix.toDoubleMatrix();
            d = flatten(m, matrix.rows(), matrix.columns());
        }
        return new SimpleMatrix(matrix.rows(), matrix.columns(), true, d);
    }

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

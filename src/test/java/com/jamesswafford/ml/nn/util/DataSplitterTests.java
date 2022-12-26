package com.jamesswafford.ml.nn.util;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.util.DataSplitter.getMiniBatch;
import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class DataSplitterTests {

    @Test
    public void splitTest1() {
        SimpleMatrix X = getX();
        SimpleMatrix Y = getY();

        Pair<SimpleMatrix, SimpleMatrix> X_Y_split = getMiniBatch(X, Y, 0, 10);
        SimpleMatrix X_split = X_Y_split.getValue0();
        assertDoubleEquals(new double[]{
                         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
                X_split.getDDRM().getData());

        SimpleMatrix Y_split = X_Y_split.getValue1();
        assertDoubleEquals(new double[]{  100, 101, 102, 103, 104, 105, 106, 107, 108, 109 },
                Y_split.getDDRM().getData());
    }

    @Test
    public void splitTest2() {
        SimpleMatrix X = getX();
        SimpleMatrix Y = getY();

        Pair<SimpleMatrix, SimpleMatrix> X_Y_split = getMiniBatch(X, Y, 0, 5);
        SimpleMatrix X_split = X_Y_split.getValue0();
        assertDoubleEquals(new double[]{  0,  1,  2,  3,  4,
                                         10, 11, 12, 13, 14,
                                         20, 21, 22, 23, 24 },
                X_split.getDDRM().getData());

        SimpleMatrix Y_split = X_Y_split.getValue1();
        assertDoubleEquals(new double[]{  100, 101, 102, 103, 104 },
                Y_split.getDDRM().getData());
    }

    @Test
    public void splitTest3() {
        SimpleMatrix X = getX();
        SimpleMatrix Y = getY();

        Pair<SimpleMatrix, SimpleMatrix> X_Y_split = getMiniBatch(X, Y, 2, 3);
        SimpleMatrix X_split = X_Y_split.getValue0();
        assertDoubleEquals(new double[]{ 6, 7, 8, 16, 17, 18, 26, 27, 28 },
                X_split.getDDRM().getData());

        SimpleMatrix Y_split = X_Y_split.getValue1();
        assertDoubleEquals(new double[]{  106, 107, 108 },
                Y_split.getDDRM().getData());
    }

    @Test
    public void splitTest4() {
        SimpleMatrix X = getX();
        SimpleMatrix Y = getY();

        Pair<SimpleMatrix, SimpleMatrix> X_Y_split = getMiniBatch(X, Y, 3, 3);
        SimpleMatrix X_split = X_Y_split.getValue0();
        assertDoubleEquals(new double[]{ 9, 19, 29 },
                X_split.getDDRM().getData());

        SimpleMatrix Y_split = X_Y_split.getValue1();
        assertDoubleEquals(new double[]{  109 },
                Y_split.getDDRM().getData());
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

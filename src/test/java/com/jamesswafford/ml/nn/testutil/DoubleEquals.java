package com.jamesswafford.ml.nn.testutil;

import org.ejml.simple.SimpleMatrix;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DoubleEquals {

    public static void assertDoubleEquals(double expected, double actual) {
        double epsilon = 0.0000001;
        assertTrue(actual >= expected - epsilon);
        assertTrue(actual <= expected + epsilon);
    }

    public static void assertDoubleEquals(List<Double> expected, List<Double> actual) {
        assertEquals(actual.size(), expected.size());
        for (int i=0;i<actual.size();i++) {
            assertDoubleEquals(expected.get(i), actual.get(i));
        }
    }

    public static void assertMatrixEquals(SimpleMatrix m1, SimpleMatrix m2) {
        assertEquals(m1.numRows(), m2.numRows());
        assertEquals(m1.numCols(), m2.numCols());
        for (int r=0;r<m1.numRows();r++) {
            for (int c=0;c<m1.numCols();c++) {
                assertDoubleEquals(m1.get(r, c), m2.get(r, c));
            }
        }
    }
}

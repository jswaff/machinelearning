package com.jamesswafford.ml.nn.testutil;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DoubleEquals {

    static double epsilon = 0.0000001;

    public static void assertDoubleEquals(double expected, double actual) {
        assertEquals(expected, actual, epsilon);
    }

    public static void assertDoubleEquals(double[] arr1, double[] arr2) {
        assertArrayEquals(arr1, arr2, epsilon);
    }
}

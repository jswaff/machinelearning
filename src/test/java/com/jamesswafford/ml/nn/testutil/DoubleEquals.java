package com.jamesswafford.ml.nn.testutil;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DoubleEquals {

    private static void assertDoubleEquals(double expected, double actual) {
        double epsilon = 0.0000001;
        assertTrue(actual >= expected - epsilon);
        assertTrue(actual <= expected + epsilon);
    }

    @Deprecated
    public static void assertDoubleEquals(double[] arr1, double[] arr2) {
        assertEquals(arr1.length, arr2.length);
        for (int i=0;i<arr1.length;i++) {
            assertDoubleEquals(arr1[i], arr2[i]);
        }
    }
}

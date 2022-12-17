package com.jamesswafford.ml.nn.testutil;

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

    public static void assertDoubleEquals(double[] arr1, double[] arr2) {
        assertEquals(arr1.length, arr2.length);
        for (int i=0;i<arr1.length;i++) {
            assertDoubleEquals(arr1[i], arr2[i]);
        }
    }
}

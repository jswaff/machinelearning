package com.jamesswafford.ml.nn.activation;

import org.junit.jupiter.api.Test;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

public class SoftmaxTests {

    private static final double epsilon = 0.00001;

    private final Softmax softmax = new Softmax();

    @Test
    //https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    public void activation1() {

        INDArray Z = Nd4j.create(new double[] { 2.0, 1.0, 0.1 }, new int[]{3,1});
        INDArray A = softmax.fn(Z);

        assertArrayEquals(new double[] { .659001139, .242432971, .09856589 }, A.toDoubleVector(), epsilon);

        // test vectorization
        Z = Nd4j.create(new double[] {2.0,0.1,1.0,1.0,1.0,1.0,0.1,2.0,0.0}, new int[]{3,3});
        A = softmax.fn(Z);
        System.out.println(A);

        assertArrayEquals(
                new double[] { .659001139, .09856589, 0.422318798,
                               .242432971, .242432971, 0.422318798,
                               .09856589,  .659001139, 0.155362403 }, ArrayUtil.flatten(A.toDoubleMatrix()), epsilon);

    }

    @Test
    //https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    public void activation2() {

        INDArray Z = Nd4j.create(new double[] { 1.1, 2.2, 0.2, -1.7 }, new int[]{4,1});
        INDArray A = softmax.fn(Z);

        assertArrayEquals(new double[] {.223636312,.67184061,.090923739,.013599338}, ArrayUtil.flatten(A.toDoubleMatrix()), epsilon);
    }

    // TODO: test cases for derivative (should yield matrix m x m)
}

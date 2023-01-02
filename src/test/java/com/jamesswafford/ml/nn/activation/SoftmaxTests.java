package com.jamesswafford.ml.nn.activation;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

public class SoftmaxTests {

    private final Softmax softmax = new Softmax();

    @Test
    //https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    public void activation1() {

        SimpleMatrix Z = new SimpleMatrix(3, 1, true, new double[] { 2.0, 1.0, 0.1 });
        SimpleMatrix A = softmax.fn(Z);

        assertDoubleEquals(new double[] { .659001139, .242432971, .09856589 }, A.getDDRM().getData());

        // test vectorization
        Z = new SimpleMatrix(3, 3, true,
                new double[] { 2.0, 0.1, 1.0,
                               1.0, 1.0, 1.0,
                               0.1, 2.0, 0.0 });
        A = softmax.fn(Z);
        A.print();

        assertDoubleEquals(
                new double[] { .659001139, .09856589, 0.422318798,
                               .242432971, .242432971, 0.422318798,
                               .09856589,  .659001139, 0.155362403 }, A.getDDRM().getData());

    }

    @Test
    //https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    public void activation2() {

        SimpleMatrix Z = new SimpleMatrix(4, 1, true, new double[] { 1.1, 2.2, 0.2, -1.7 });
        SimpleMatrix A = softmax.fn(Z);

        assertDoubleEquals(new double[] { .223636312,  .67184061, .090923739, .013599338}, A.getDDRM().getData());
    }

    // TODO: test cases for derivative (should yield matrix m x m)
}

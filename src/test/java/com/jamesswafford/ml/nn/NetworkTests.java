package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.Sigmoid;
import com.jamesswafford.ml.nn.cost.MSE;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkTests {

    @Test
    public void andGate() {
        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(new Layer(1, new Sigmoid())))
                .costFunction(new MSE())
                .build();

        network.initialize();

        SimpleMatrix X = new SimpleMatrix(2, 4, true,
                new double[]{ 0, 0, 1, 1,
                              0, 1, 1, 0 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 0, 1, 0 });
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 1000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

    @Test
    public void xorGate() {
        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(
                        new Layer(2, new Sigmoid()),
                        new Layer(1, new Sigmoid())
                ))
                .costFunction(new MSE())
                .build();

        network.initialize();

        SimpleMatrix X = new SimpleMatrix(2, 4, true,
                new double[]{ 0, 0, 1, 1,
                              0, 1, 1, 0 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 1, 0, 1});
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 1000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

    @Test
    public void booleanExpression() {
        // this test models the boolean expression
        // AND( NOT(AND(A,B)), NOT(OR(A,B)), C)
        // This expression is equivalent to the logic gate found here:
        // https://www.watelectronics.com/what-is-a-combinational-logic-circuit-types-and-applications/

        Network network = Network.builder()
                .numInputUnits(3)
                .layers(List.of(
                        new Layer(2, new Sigmoid()),
                        new Layer(1, new Sigmoid())
                ))
                .costFunction(new MSE())
                .build();

        network.initialize();

        SimpleMatrix X = new SimpleMatrix(3, 8, true,
                new double[]{ 0,0,0,0,1,1,1,1,
                              0,0,1,1,0,0,1,1,
                              0,1,0,1,0,1,0,1 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 8, true,
                new double[]{ 0,0,0,0,1,0,0,0 });
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 100000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

    @Test
    public void threeHot() {
        // given 4 inputs, exactly three must be active
        Network network = Network.builder()
                .numInputUnits(4)
                .layers(List.of(
                        new Layer(3, new Sigmoid()),
                        new Layer(3, new Sigmoid()),
                        new Layer(1, new Sigmoid())
                ))
                .costFunction(new MSE())
                .build();

        network.initialize();

        SimpleMatrix X = new SimpleMatrix(4, 16, true,
                new double[]{ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,
                              0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,
                              0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,
                              0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 16, true,
                new double[]{ 0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0 });
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 100000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

}

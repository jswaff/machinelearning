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

        // input matrix is 2 (features) x 4 (training examples)
        // 0 0 1 1
        // 0 1 0 1
        SimpleMatrix X = new SimpleMatrix(2, 4, false,
                new double[]{ 0, 0, 1, 1, 0, 1, 1, 0 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, false,
                new double[]{ 0, 1, 0, 0});
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 1000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

    @Test
    public void orGate() {
        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(new Layer(1, new Sigmoid())))
                .costFunction(new MSE())
                .build();

        network.initialize();

        // input matrix is 2 (features) x 4 (training examples)
        // 0 0 1 1
        // 0 1 0 1
        SimpleMatrix X = new SimpleMatrix(2, 4, false,
                new double[]{ 0, 0, 1, 1, 0, 1, 1, 0 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, false,
                new double[]{ 0, 1, 1, 1});
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

        // input matrix is 2 (features) x 4 (training examples)
        // 0 0 1 1
        // 0 1 0 1
        SimpleMatrix X = new SimpleMatrix(2, 4, false,
                new double[]{ 0, 0, 1, 1, 0, 1, 1, 0 });
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, false,
                new double[]{ 0, 0, 1, 1});
        Y.print();

        // initial cost

        // train the network
        network.train(X, Y, 1000);

        SimpleMatrix P = network.predict(X);
        P.print();

        // the cost should be close to 0

    }

}

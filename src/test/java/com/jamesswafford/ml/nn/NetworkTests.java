package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.Sigmoid;
import com.jamesswafford.ml.nn.cost.MSE;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.jamesswafford.ml.nn.testutil.DoubleEquals.*;

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
        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 0, 1, 0 });

        train(network, X, Y);
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

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 1, 0, 1});

        train(network, X, Y);
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

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 8, true,
                new double[]{ 0,0,0,0,1,0,0,0 });

        train(network, X, Y);
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

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 16, true,
                new double[]{ 0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0 });

        train(network, X, Y);
    }

    @Test
    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    public void exampleFromMattMazur() {
        Network network = buildExampleNetworkFromMM();
        Layer hidden = network.getLayers().get(0);
        Layer output = network.getLayers().get(1);

        SimpleMatrix X = new SimpleMatrix(2, 1, true, new double[] {0.05, 0.10});
        SimpleMatrix Y = new SimpleMatrix(2, 1, true, new double[] {0.01, 0.99});

        SimpleMatrix P = network.predict(X);

        assertDoubleEquals(0.3775, hidden.getZ().get(0, 0));
        assertDoubleEquals(new double[] { .593269992, .596884378}, hidden.getA().getDDRM().getData());

        assertDoubleEquals(1.10590597, output.getZ().get(0, 0));
        assertDoubleEquals(new double[]{ .75136507, .772928465 }, output.getA().getDDRM().getData());
        assertDoubleEquals(new double[]{ .75136507, .772928465 }, P.getDDRM().getData());

        // test the initial cost
        double cost = network.cost(P, Y);
        assertDoubleEquals(0.2983711087600027, cost);

        // do one forward and backward pass
        network.train(X, Y, 1, 1, 0.5, null, null);

        assertDoubleEquals(new double[]{.149780716, .199561432, .249751144,  .299502287},
                hidden.getWeights().getDDRM().getData());

        assertDoubleEquals(new double[]{.35891648, .408666186, .51130127, .561370121},
                output.getWeights().getDDRM().getData());

        // the cost after updating weights does not match Matt's, but that's because he does not update the
        // bias terms.  Comments confirm the correct value is 0.28047144679143016
        SimpleMatrix P2 = network.predict(X);
        double cost2 = network.cost(P2, Y);
        assertDoubleEquals(0.28047144679143016, cost2);

        // Matt gives an error of 3.510187782978859E-5 after 10,000 epochs, but again this is without updating biases
        // with bias updates the cost is 2.4475622359322466E-6
        Network network2 = buildExampleNetworkFromMM();
        network2.train(X, Y, 10000, 1, 0.5, null, null);
        assertDoubleEquals(2.4475622359322466E-6, network2.cost(network2.predict(X), Y));
    }

    private void train(Network network, SimpleMatrix X, SimpleMatrix Y) {

        // train the network
        network.train(X, Y, 100000, X.numCols(), 3.0, null, null);

        Y.print();

        SimpleMatrix P = network.predict(X);
        P.print();
        double cost = network.cost(P, Y);
        System.out.println("cost: " + cost);
    }

    private Network buildExampleNetworkFromMM() {
        Layer hidden = new Layer(2, new Sigmoid());

        Layer output = new Layer(2, new Sigmoid());

        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(
                        hidden,
                        output
                ))
                .costFunction(new MSE())
                .build();
        network.initialize();

        // override the weights and biases with fixed values for testing
        hidden.setWeight(0, 0, 0.15);
        hidden.setWeight(0, 1, 0.20);
        hidden.setBias(0, 0.35);

        hidden.setWeight(1, 0, 0.25);
        hidden.setWeight(1, 1, 0.30);
        hidden.setBias(1, 0.35);

        output.initialize(2);
        output.setWeight(0, 0, 0.4);
        output.setWeight(0, 1, 0.45);
        output.setBias(0, 0.60);

        output.setWeight(1, 0, 0.5);
        output.setWeight(1, 1, 0.55);
        output.setBias(1, 0.60);

        return network;
    }

}

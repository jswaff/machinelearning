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
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 0, 1, 0 });
        Y.print();

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
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 4, true,
                new double[]{ 0, 1, 0, 1});
        Y.print();

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
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 8, true,
                new double[]{ 0,0,0,0,1,0,0,0 });
        Y.print();

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
        X.print();

        // labels
        SimpleMatrix Y = new SimpleMatrix(1, 16, true,
                new double[]{ 0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0 });
        Y.print();

        train(network, X, Y);
    }

    @Test
    // example problem from TDS article by Tobias Hill
    // https://towardsdatascience.com/part-1-a-neural-network-from-scratch-foundation-e2d119df0f40
    public void exampleFromTDS() {

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

        hidden.setWeight(0, 0, 0.3);
        hidden.setWeight(0, 1, -0.4);
        hidden.setBias(0, 0.25);

        hidden.setWeight(1, 0, 0.2);
        hidden.setWeight(1, 1, 0.6);
        hidden.setBias(1, 0.45);

        output.initialize(2);
        output.setWeight(0, 0, 0.7);
        output.setWeight(0, 1, 0.5);
        output.setBias(0, 0.15);

        output.setWeight(1, 0, -0.3);
        output.setWeight(1, 1, -0.1);
        output.setBias(1, 0.35);

        SimpleMatrix X = new SimpleMatrix(2, 1, true, new double[] {2, 3});
        SimpleMatrix Y = new SimpleMatrix(2, 1, true, new double[] {1, 0.2});

        SimpleMatrix P = network.predict(X);
        assertDoubleEquals(0.712257432, P.get(0,0));
        assertDoubleEquals(0.533097573, P.get(1, 0));

        double cost = network.cost(P, Y);
        System.out.println("cost: " + cost);
        assertDoubleEquals(0.1937497789, cost);

        System.out.println("dZdW: " + output.getX().get(0,0)); // Oh1: 0.41338242108266987
        System.out.println("Z: ");  // [ .90637319  .132584175 ]
        output.getZ().print();
        // In the TDS article, Toby shows 0.220793265 here.  It appears he is actually calculating g'(A), not g'(Z)
        System.out.println("dAdZ: " + output.calculateZPrime()); // Z': 0.220793265

        // train
        network.train(X, Y, 1, 1, 1.0, null, null);
        SimpleMatrix P2 = network.predict(X);
        //P2.print();    // expect: y = [0.719269360605435 0.524309343003261]


        double cost2 = network.cost(P2, Y);
        // expect: 0.1839862418540884.
        //System.out.println("cost2: " + cost2);
    }

    private void train(Network network, SimpleMatrix X, SimpleMatrix Y) {
        // initial cost
        SimpleMatrix P1 = network.predict(X);
        double cost = network.cost(P1, Y);
        System.out.println("initial cost: " + cost);

        // train the network
        network.train(X, Y, 100000, X.numCols(), 1.0, null, null);

        // cost after training
        SimpleMatrix P2 = network.predict(X);
        P2.print();
        cost = network.cost(P2, Y);
        System.out.println("final cost: " + cost);
    }

}

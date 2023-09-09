package com.jamesswafford.ml.nn;

import com.google.gson.Gson;
import com.jamesswafford.ml.nn.activation.Sigmoid;
import com.jamesswafford.ml.nn.cost.MSE;
import org.junit.jupiter.api.Test;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkTests {

    private static final double epsilon = 0.00001;

    @Test
    public void andGate() {
        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(new Layer(1, Sigmoid.INSTANCE)))
                .costFunction(MSE.INSTANCE)
                .build();

        network.initialize();

        INDArray X = Nd4j.create(new double[][] {{0,0,1,1}, {0,1,1,0}});
        INDArray Y = Nd4j.create(new double[]{0,0,1,0}, new int[]{1,4});

        train(network, X, Y);
    }

    @Test
    public void xorGate() {
        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(
                        new Layer(2, Sigmoid.INSTANCE),
                        new Layer(1, Sigmoid.INSTANCE)
                ))
                .costFunction(MSE.INSTANCE)
                .build();

        network.initialize();

        INDArray X = Nd4j.create(new double[][] {{0,0,1,1},{0,1,1,0}});
        INDArray Y = Nd4j.create(new double[]{0,1,0,1}, new int[]{1,4});

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
                        new Layer(2, Sigmoid.INSTANCE),
                        new Layer(1, Sigmoid.INSTANCE)
                ))
                .costFunction(MSE.INSTANCE)
                .build();

        network.initialize();

        INDArray X = Nd4j.create(new double[][] {
                {0,0,0,0,1,1,1,1},
                {0,0,1,1,0,0,1,1},
                {0,1,0,1,0,1,0,1}
        });
        INDArray Y = Nd4j.create(new double[]{ 0,0,0,0,1,0,0,0 }, new int[]{1,8});

        train(network, X, Y);
    }

    @Test
    public void threeHot() {
        // given 4 inputs, exactly three must be active
        Network network = Network.builder()
                .numInputUnits(4)
                .layers(List.of(
                        new Layer(3, Sigmoid.INSTANCE),
                        new Layer(3, Sigmoid.INSTANCE),
                        new Layer(1, Sigmoid.INSTANCE)
                ))
                .costFunction(MSE.INSTANCE)
                .build();

        network.initialize();

        INDArray X = Nd4j.create(new double[][] {
                {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
                {0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1},
                {0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1},
                {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1}
        }) ;

        INDArray Y = Nd4j.create(new double[]{ 0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0 }, new int[]{1,16});

        train(network, X, Y);
    }

    @Test
    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    public void exampleFromMattMazur() {
        Network network = buildExampleNetworkFromMM();
        Layer hidden = network.getLayers().get(0);
        Layer output = network.getLayers().get(1);

        INDArray X = Nd4j.create(new double[]{0.05,0.10}, new int[]{2,1});
        INDArray Y = Nd4j.create(new double[]{0.01,0.99}, new int[]{2,1});

        INDArray P = network.predict(X);

        //assertEquals(0.3775, hidden.getZ().getDouble(0, 0), epsilon);
        assertArrayEquals(new double[] { .593269992, .596884378}, hidden.getA().toDoubleVector(), epsilon);

        //assertEquals(1.10590597, output.getZ().getDouble(0, 0), epsilon);
        assertArrayEquals(new double[]{ .75136507, .772928465 }, output.getA().toDoubleVector(), epsilon);
        assertArrayEquals(new double[]{ .75136507, .772928465 }, P.toDoubleVector(), epsilon);

        // test the initial cost
        double cost = network.cost(P, Y);
        assertEquals(0.2983711087600027, cost, epsilon);

        // do one forward and backward pass
        network.train(X, Y, 1, 1, 0.5, null, null);

        assertArrayEquals(new double[]{.149780716, .199561432, .249751144,  .299502287},
                ArrayUtil.flatten(hidden.getWeights().toDoubleMatrix()), epsilon);

        assertArrayEquals(new double[]{.35891648, .408666186, .51130127, .561370121},
                ArrayUtil.flatten(output.getWeights().toDoubleMatrix()), epsilon);

        // the cost after updating weights does not match Matt's, but that's because he does not update the
        // bias terms.  Comments confirm the correct value is 0.28047144679143016
        INDArray P2 = network.predict(X);
        double cost2 = network.cost(P2, Y);
        assertEquals(0.28047144679143016, cost2, epsilon);

        // Matt gives an error of 3.510187782978859E-5 after 10,000 epochs, but again this is without updating biases
        // with bias updates the cost is 2.4475622359322466E-6
        Network network2 = buildExampleNetworkFromMM();
        network2.train(X, Y, 10000, 1, 0.5, null, null);
        assertEquals(2.4475622359322466E-6, network2.cost(network2.predict(X), Y), epsilon);
    }

    @Test
    public void toAndFromState() {
        Network network = buildExampleNetworkFromMM();
        Network.NetworkState state = network.getState();
        assertEquals(2, state.getNumInputUnits());
        assertEquals(2, state.getLayers().length); // layers "toState" tested in LayerTests
        assertEquals("mse", state.getCostFunction());

        Network network2 = Network.fromState(state);
        assertEquals(2, network2.getNumInputUnits());
        assertEquals(MSE.INSTANCE, network2.getCostFunction());
        assertEquals(2, network2.getLayers().size()); // layers "fromState" tested in LayerTests
    }

    @Test
    public void toAndFromJson() {
        Network network = buildExampleNetworkFromMM();
        String json = network.toJson();
        Network.NetworkState state = new Gson().fromJson(json, Network.NetworkState.class);
        assertEquals(network.getState(), state);

        Network network2 = Network.fromJson(json);
        assertEquals(network.getNumInputUnits(), network2.getNumInputUnits());
        assertEquals(network.getCostFunction(), network2.getCostFunction());
        assertEquals(network.getLayers().size(), network2.getLayers().size());
        for (int i=0;i<network.getLayers().size();i++) {
            Layer layer1 = network.getLayers().get(i);
            Layer layer2 = network2.getLayers().get(i);
            assertEquals(layer1.getNumUnits(), layer2.getNumUnits());
            assertEquals(layer1.getActivationFunction(), layer2.getActivationFunction());
            assertArrayEquals(layer1.getWeights().toDoubleMatrix(), layer2.getWeights().toDoubleMatrix());
            assertArrayEquals(layer1.getBiases().toDoubleVector(), layer2.getBiases().toDoubleVector());
        }
    }

    private void train(Network network, INDArray X, INDArray Y) {

        // train the network
        network.train(X, Y, 100000, X.columns(), 3.0, null, null);

        //Y.print();
        System.out.println(Y);

        INDArray P = network.predict(X);
        //P.print();
        System.out.println(P);
        double cost = network.cost(P, Y);
        System.out.println("cost: " + cost);
    }

    private Network buildExampleNetworkFromMM() {
        Layer hidden = new Layer(2, Sigmoid.INSTANCE);

        Layer output = new Layer(2, Sigmoid.INSTANCE);

        Network network = Network.builder()
                .numInputUnits(2)
                .layers(List.of(
                        hidden,
                        output
                ))
                .costFunction(MSE.INSTANCE)
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

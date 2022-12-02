package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.Identity;
import com.jamesswafford.ml.nn.cost.MSE;
import org.junit.jupiter.api.Test;

import java.util.List;

public class NetworkTests {

    @Test
    public void oneLayer() {
        Network network = Network.builder()
                .numInputUnits(1)
                .layers(List.of(new Layer(1, new Identity())))
                .costFunction(new MSE())
                .build();

        network.initialize();
    }

}

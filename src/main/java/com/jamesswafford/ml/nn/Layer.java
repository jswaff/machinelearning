package com.jamesswafford.ml.nn;

import com.jamesswafford.ml.nn.activation.ActivationFunction;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.ejml.simple.SimpleMatrix;

@RequiredArgsConstructor
public class Layer {

    @Getter
    private final int numUnits;
    private final ActivationFunction activationFunction;

    private SimpleMatrix w;
    private SimpleMatrix b;

    public void initialize(int numUnitsPreviousLayer) {
        w = new SimpleMatrix(numUnits, numUnitsPreviousLayer);
        for (int r=0;r<numUnits;r++) {
            for (int c=0;c<numUnitsPreviousLayer;c++) {
                w.set(r, c, Math.random());
            }
        }
        b = new SimpleMatrix(numUnits, 1);
        for (int r=0;r<numUnits;r++) {
            b.set(r, 0, 0.0);
        }
    }

    public Double getWeight(int unit, int prevUnit) {
        return w.get(unit, prevUnit);
    }

    public Double getBias(int unit) {
        return b.get(unit, 0);
    }

    public SimpleMatrix linearForward(SimpleMatrix A) {
        SimpleMatrix Z = new SimpleMatrix(A);
        //TODO Z = np.dot(W, A) + b
        return Z;
    }

    public SimpleMatrix activationForward(SimpleMatrix A) {
        SimpleMatrix Z = linearForward(A);
        SimpleMatrix myA = new SimpleMatrix(Z);
        // TODO - apply activation function
        return myA;
    }
}

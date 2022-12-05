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

    public void setWeight(int unit, int prevUnit, Double val) {
        w.set(unit, prevUnit, val);
    }

    public Double getBias(int unit) {
        return b.get(unit, 0);
    }

    public void setBias(int unit, Double val) {
        b.set(unit, 0, val);
    }

    public SimpleMatrix linearForward(SimpleMatrix A) {
        return w.mult(A).plus(b);
    }

    public SimpleMatrix activationForward(SimpleMatrix prevA) {
        SimpleMatrix Z = linearForward(prevA);
        SimpleMatrix A = new SimpleMatrix(Z);
        // TODO - apply activation function
        return A;
    }
}

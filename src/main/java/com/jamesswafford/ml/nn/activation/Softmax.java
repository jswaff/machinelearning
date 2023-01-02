package com.jamesswafford.ml.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class Softmax {

    public SimpleMatrix fn(SimpleMatrix Z) {
        assert(Z.numCols()==1);
        SimpleMatrix A = new SimpleMatrix(Z.numRows(), 1);

        double sum = 0;
        for (int r=0;r<Z.numRows();r++) {
            double v = Math.exp(Z.get(r, 0));
            A.set(r, 0, v);
            sum += v;
        }

        return A.divide(sum);
    }

    // reference: https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
    public SimpleMatrix dFn(SimpleMatrix X) {
        assert(X.numCols()==1);
        int n = X.numRows();

        SimpleMatrix P = fn(X);

        SimpleMatrix out = new SimpleMatrix(n, n);
        for (int i=0;i<n;i++) {
            for (int j=0;j<n;j++) {
                double d = i==j ? 1.0 : 0.0;
                double p_i = P.get(i, 0);
                double p_j = P.get(j, 0);
                out.set(i, j, p_i * (d - p_j));
            }
        }

        return out;
    }

}

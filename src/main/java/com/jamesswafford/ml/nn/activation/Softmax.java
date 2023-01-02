package com.jamesswafford.ml.nn.activation;

import org.ejml.simple.SimpleMatrix;

// this class is a work in progress.  the derivative function needs to be vectorized and tested, then
// fit into the Activation interface somehow.  Perhaps changing the interface to use SimpleMatrix as input
// and output rather than a single element
public class Softmax {

    public SimpleMatrix fn(SimpleMatrix Z) {

        SimpleMatrix A = new SimpleMatrix(Z.numRows(), Z.numCols());

        double[] sum = new double[Z.numCols()];
        for (int r=0;r<Z.numRows();r++) {
            for (int c=0;c<Z.numCols();c++) {
                double v = Math.exp(Z.get(r, c));
                A.set(r, c, v);
                sum[c] += v;
            }
        }

        // normalize
        for (int r=0;r<Z.numRows();r++) {
            for (int c=0;c<Z.numCols();c++) {
                A.set(r, c, A.get(r, c) / sum[c]);
            }
        }

        return A;
    }

    // reference: https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
    // TODO: vectorize
    // A has one row per output neuron and one column per sample
    // each column should produce a mxm matrix
    public SimpleMatrix dFn(SimpleMatrix A) {
        assert(A.numCols()==1);
        int n = A.numRows();

        SimpleMatrix P = fn(A);

        SimpleMatrix out = new SimpleMatrix(n, n);
        for (int r=0;r<n;r++) {
            for (int c=0;c<n;c++) {
                double d = r==c ? 1.0 : 0.0;
                double p_r = P.get(r, 0);
                double p_c = P.get(c, 0);
                out.set(r, c, p_r * (d - p_c));
            }
        }

        return out;
    }

}

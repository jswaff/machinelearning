package com.jamesswafford.ml.nn.activation;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

// this class is a work in progress.  the derivative function needs to be vectorized and tested, then
// fit into the Activation interface somehow.  Perhaps changing the interface to use SimpleMatrix as input
// and output rather than a single element
public class Softmax {

    public INDArray fn(INDArray Z) {

        INDArray A = Nd4j.create(DataType.DOUBLE, Z.rows(), Z.columns());

        double[] sum = new double[Z.columns()];
        for (int r=0;r<Z.rows();r++) {
            for (int c=0;c<Z.columns();c++) {
                double v = Math.exp(Z.getDouble(r, c));
                A.putScalar(r, c, v);
                sum[c] += v;
            }
        }

        // normalize
        for (int r=0;r<Z.rows();r++) {
            for (int c=0;c<Z.columns();c++) {
                A.putScalar(r, c, A.getDouble(r, c) / sum[c]);
            }
        }

        return A;
    }

    // reference: https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
    // TODO: vectorize
    // A has one row per output neuron and one column per sample
    // each column should produce a mxm matrix
    public INDArray dFn(INDArray A) {
        assert(A.columns()==1);
        int n = A.rows();

        INDArray P = fn(A);

        INDArray out = Nd4j.zeros(n, n);
        for (int r=0;r<n;r++) {
            for (int c=0;c<n;c++) {
                double d = r==c ? 1.0 : 0.0;
                double p_r = P.getDouble(r, 0);
                double p_c = P.getDouble(c, 0);
                out.putScalar(r, c, p_r * (d - p_c));
            }
        }

        return out;
    }

}

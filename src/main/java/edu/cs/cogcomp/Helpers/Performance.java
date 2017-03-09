package edu.cs.cogcomp.Helpers;

/**
 * Created by Snigdha on 2/20/17.
 */
public class Performance {
    // source performance measures
    double source_Acc = 0.0;
    double source_F = 0.0;
    double source_Roc = 0.0;

    // target performance measures
    double target_Acc = 0.0;
    double target_F = 0.0;
    double target_Roc = 0.0;

    public Performance(double[] performance) {
        source_Acc = performance[0];
        source_F = performance[1];
        source_Roc = performance[2];
        target_Acc = performance[3];
        target_F = performance[4];
        target_Roc = performance[5];
    }
}

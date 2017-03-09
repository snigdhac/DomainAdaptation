package edu.cs.cogcomp.Helpers;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.util.Pair;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by Snigdha on 2/20/17.
 */
public class Utils {
    static int seedForRandom = 234; // fixing the seed helps in getting the same random result everytime.

    /*
     * randomly split a dataset of Instances into train and test sets.
     * Size of train set is x% of the input dataset where x=train_split
     */
    public static Pair<Instances, Instances> trainTestSplit(Instances data, double train_split, int seed) {
        data.randomize(new Random(seed));
        int trainSize = (int) Math.round(data.numInstances() * train_split);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        return new Pair<Instances, Instances>(train, test);

    }

    /*
     * Upsample the minority class in a dataset by repetition.
     * NOTE: the number of instances in the two classes will NOT BE exactly (but roughly) equal in the upsampled dataset
     */
    public static Instances upsample(Instances data, String class_minority, String class_majority) {
        Instances upsampled_data = new Instances(data);

        Pair<Integer, Integer> class_Dist = getClassDist(data, class_majority, class_minority);
        int frac = (int) ((double)class_Dist.getSecond()/class_Dist.getFirst());

        for(int x=0;x<data.numInstances();x++){
            Instance point = data.instance(x);
            String label = data.classAttribute().value((int) point.classValue());
            if(label.equals(class_minority))
                for(int i=0;i<frac;i++)
                    upsampled_data.add(point);

            // sanity check
            if(!label.equals(class_majority) && !label.equals(class_minority)){
                System.err.println("Found unexpected class="+label + " I expect to only see "+class_majority+" or "+class_minority);
                System.err.println("Don't know how to upsample now. Exiting");
                System.exit(-1);
            }
        }

        // Uncomment the following lines to print the upsampled dataset's class distribution
//        class_Dist = getClassDist(upsampled_data, class_majority, class_minority);
//        System.out.println("Class Distribution in the upsampled dataset is= "+class_Dist.getFirst()+"...."+class_Dist.getSecond()+". This was created using a fraction of "+frac);

        upsampled_data.randomize(new Random(seedForRandom));
        return upsampled_data;
    }

    /*
     * Helper function.
     * Counts the number of instances belonging to the two input classes in a given dataset
     */
    private static Pair<Integer,Integer> getClassDist(Instances data, String class_majority, String class_minority) {
        int min_count=0, maj_count=0;
        for(int x=0;x<data.numInstances();x++){
            Instance point = data.instance(x);
            String classIndex = data.classAttribute().value((int) point.classValue());
            if(classIndex.equals(class_majority))
                maj_count++;
            else if(classIndex.equals(class_minority))
                min_count++;
        }
        return new Pair<>(min_count,maj_count);
    }

    /*
     * This function takes as input an array list of 'Performance' and print median(or mean) of individual performance measures
     * To get means instead of medians, simply replace Utils.getMedian() with Utils.getMean() while printing
     */
    public static void printMedians(ArrayList<Performance> arrayListPerformances) {
        // Form independent arraylists from individual performance measures, then calculate the mean/median of the arraylist.
        ArrayList<Double> src_acc = new ArrayList<>();
        ArrayList<Double> src_f = new ArrayList<>();
        ArrayList<Double> src_roc = new ArrayList<>();
        ArrayList<Double> tgt_acc = new ArrayList<>();
        ArrayList<Double> tgt_f = new ArrayList<>();
        ArrayList<Double> tgt_roc = new ArrayList<>();
        for(Performance p:arrayListPerformances){
            src_acc.add(p.source_Acc);
            src_f.add(p.source_F);
            src_roc.add(p.source_Roc);
            tgt_acc.add(p.target_Acc);
            tgt_f.add(p.target_F);
            tgt_roc.add(p.target_Roc);
        }

        // For printing just numbers helps in copy pasting
//        System.out.printf("%.2f\n", Utils.getMedian(src_acc));
//        System.out.printf("%.2f\n", Utils.getMedian(src_f));
//        System.out.printf("%.2f\n", Utils.getMedian(src_roc));
//        System.out.printf("%.2f\n", Utils.getMedian(tgt_acc));
//        System.out.printf("%.2f\n", Utils.getMedian(tgt_f));
//        System.out.printf("%.2f\n", Utils.getMedian(tgt_roc));
//        System.out.println();

        System.out.printf("Source Median Accuracy = %.2f\n", Utils.getMedian(src_acc));
        System.out.printf("Source Median F1 score = %.2f\n", Utils.getMedian(src_f));
        System.out.printf("Source Median Roc      = %.2f\n", Utils.getMedian(src_roc));
        System.out.printf("Target Median accuracy = %.2f\n", Utils.getMedian(tgt_acc));
        System.out.printf("Target Median F1 score = %.2f\n", Utils.getMedian(tgt_f));
        System.out.printf("Target Median Roc      = %.2f\n", Utils.getMedian(tgt_roc));
        System.out.println();
    }

    /*
     * Computes median value of an arraylist of doubles
     */
    private static double getMedian(ArrayList<Double> m) {
        if(m==null || m.size()==0 )
            return Double.NaN;
        Collections.sort(m);
        int middle = m.size()/2;
        if (m.size()%2 == 1) {
            return m.get(middle);
        } else {
            return (m.get(middle-1) + m.get(middle)) / 2.0;
        }
    }

    /*
     * Computes mean value of an arraylist of doubles
     */
    private static double getMean(ArrayList<Double> m) {
        double sum = 0;
        for (int i = 0; i < m.size(); i++) {
            sum += m.get(i);
        }
        return sum / m.size();
    }

    /*
     * pretty print a one dimensional double array
     */
    public static void printArray(double [] arr){
        System.out.print(arr[0]);
        for(int i=1; i< arr.length; i++){
            System.out.print(" " +arr[i]);
        }
        System.out.println();
    }

    //Begin: Ziheng's helper functions
    /*
     * Helper function that calculates distances from all test set instances to a training set's centroid
     * And then mark the testing instances have top k percent distances
     */
    public static double [] centroidDistanceTest(Instances trainSet, Instances testSet){

        int num_testInst = testSet.numInstances();
        int k = 0;
        EuclideanDistance distanceComputer = new EuclideanDistance();

        //compute the centroid of the trainSet data
        int numAtt = trainSet.numAttributes();
        double [] meanVector = new double [numAtt];
        int attIdx = 0;
        for(attIdx = 0; attIdx < numAtt; attIdx++){
            meanVector[attIdx] = trainSet.meanOrMode(attIdx);
        }

        //compute all the distances
        int instIdx = 0;
        double [] distances = new double [num_testInst];
        //System.out.println("number of train att " + trainSet.numAttributes());
        //System.out.println("number of test att "  + testSet.numAttributes());
        //System.out.println("number of inst " + num_testInst);

        for(instIdx = 0; instIdx < num_testInst; instIdx++){
            double [] cur_inst = testSet.instance(instIdx).toDoubleArray();
            double cur_dist = distanceComputer.compute(cur_inst, meanVector);
            distances[instIdx] = Math.abs(cur_dist);
            //System.out.println("CURR DIST" + distances[instIdx]);

        }

        //kth largest distance
        double kth_dist = quickselect(distances, 1);
        //System.out.println("MAX DIST" + kth_dist);


        //mark the result array
        for(int i = 0; i < num_testInst; i++){
            distances[i] = distances[i]/kth_dist;
            if(distances[i] > 1){
                System.out.println("!!!!WRONG MAX DISTANCE CALCULATION!!!!");
            }
        }
        return distances;
    }

    /*
     * A quickselect implementation that can be used to find the distance threshold
     */
    public static double quickselect(double[] G, int k) {
        return quickselect(G, 0, G.length - 1, k - 1);
    }

    private static double quickselect(double[] G, int first, int last, int k) {
        if (first <= last) {
            int pivot = partition(G, first, last);
            if (pivot == k) {
                return G[k];
            }
            if (pivot > k) {
                return quickselect(G, first, pivot - 1, k);
            }
            return quickselect(G, pivot + 1, last, k);
        }
        return Integer.MIN_VALUE;
    }

    private static int partition(double[] G, int first, int last) {
        int pivot = first + new Random(seedForRandom).nextInt(last - first + 1);
        swap(G, last, pivot);
        for (int i = first; i < last; i++) {
            if (G[i] > G[last]) {
                swap(G, i, first);
                first++;
            }
        }
        swap(G, first, last);
        return first;
    }

    private static void swap(double[] G, int x, int y) {
        double tmp = G[x];
        G[x] = G[y];
        G[y] = tmp;
    }


}

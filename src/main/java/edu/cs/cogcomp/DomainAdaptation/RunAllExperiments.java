package edu.cs.cogcomp.DomainAdaptation;

/**
 * Created by Snigdha on 2/17/17.
 */

import edu.cs.cogcomp.Helpers.Performance;
import edu.cs.cogcomp.Helpers.Utils;
import edu.cs.cogcomp.Helpers.sampler;
import org.apache.commons.math3.util.Pair;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.HashSet;

public class RunAllExperiments {

    private static int seed = 7815; // fixing the seed helps in getting the same random result everytime.

    private static Logistic model = null;
//    private static LibLINEAR model = null;

    /*
     * This code randomly splits data into train and test sets and runs the Domain Adaptation method + baselines on it.
     * This process is repeated 'numSplits' number of times
     */
    public static void main(String[] args) throws Exception {
        /********************** Parameters *******************/
        int numSplits = 5; // number of times to redo experiment
        double train_split = 0.6; // percent of data to belong to trainset
        boolean upsampleTrainOrTest = true; // Do you want to upsample train or test set?
        String taskName = "IP"; // name of task (helps in determining data location). Can be "IP" or "CD" ONLY.

        int max_iter_DiAd = 1000; // parameter for Diad-radius
        double radius = 0.1; // parameter for Diad-radius
        int numInstancesToSamplePerIter = 10; // parameter for Diad-random. Number of instances to be randomly sampled per iteration

        // uncomment the following line if you want to use Weka's logistic regression (Also check line 26-27)
        model = new Logistic();

        // uncomment the following three lines if you want to use Liblinear (Also check line 26-27)
//        model = new LibLINEAR();
//        model.setOptions(weka.core.Utils.splitOptions(" -S 0 -C 1.0 -E 0.001 -B 1.0 -L 0.1 -I 1000 -W 1"));
//        model.setProbabilityEstimates(true);

        /***************************************************/

        String currentDir = System.getProperty("user.dir");
        String src_domain=null, tgt_domain=null, filename_src=null, filename_tgt=null;

        if(taskName.toLowerCase().equals("ip")) {
            //courseNames for instructor intervention prediction task
            src_domain = "WCR";
            tgt_domain = "GHC";
            filename_src = currentDir + "/data/IP/" + src_domain + "_original.arff";
            filename_tgt = currentDir + "/data/IP/" + tgt_domain + "_original.arff";
        }
        else if(taskName.toLowerCase().equals("cd")){
            //courseNames for confusion detection task
            src_domain = "HS";
            tgt_domain = "EDU";
            filename_src = currentDir + "/data/CD/features_" + src_domain + "_all_by_" + src_domain + "_vocab.arff";
            filename_tgt = currentDir + "/data/CD/features_" + tgt_domain + "_all_by_" + src_domain + "_vocab.arff";
        }
        else{
            System.err.println("I don't understand task name. It can be 'CD' or 'IP'. Exiting");
            System.exit(-1);
        }

        /********************** Start code **********************/
        System.out.println("Source="+filename_src);
        System.out.println("Target="+filename_tgt);

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename_src);
        Instances src_data = source.getDataSet();
        src_data.setClassIndex(src_data.numAttributes() - 2);

        ConverterUtils.DataSource target = new ConverterUtils.DataSource(filename_tgt);
        Instances tgt_data = target.getDataSet();
        tgt_data.setClassIndex(tgt_data.numAttributes() - 2);

        ArrayList<Performance> srcOnlyPerformance = new ArrayList<Performance>();
        ArrayList<Performance> tgtOnlyPerformance = new ArrayList<Performance>();
        ArrayList<Performance> diadPerformance = new ArrayList<Performance>();
        ArrayList<Performance> diadRandomPerformance = new ArrayList<Performance>();
        ArrayList<Performance> sPlusTPerformance = new ArrayList<Performance>();
        ArrayList<Performance> fePerformance = new ArrayList<Performance>();

        for (int i = 1; i <= numSplits; i++) {

            seed+=213; // seed changed so that we have a new test split everytime. For complete randomization, simply remove seed

            // Get data split for source
            Pair<Instances, Instances> pair = Utils.trainTestSplit(src_data, train_split, seed);
            Instances upsampled_src_train = pair.getFirst();
            if(upsampleTrainOrTest) // upsample the train set
                upsampled_src_train = Utils.upsample(pair.getFirst(), "1", "0");
            Instances src_test = pair.getSecond();
//            if(upsampleTrainAndTest) // commented because we don't want to upsample test set
//                src_test = Utils.upsample(src_test, "1", "0");

            // Get data split for target
            pair = Utils.trainTestSplit(tgt_data, train_split, seed);
            Instances upsampled_tgt_train = pair.getFirst();
            if(upsampleTrainOrTest) // upsample the train set
                upsampled_tgt_train = Utils.upsample(pair.getFirst(), "1", "0");
            Instances tgt_test = pair.getSecond();
//            if(upsampleTrainAndTest) // commented because we don't want to upsample test set
//                tgt_test = Utils.upsample(tgt_test, "1", "0");

            // Src only classifier
            System.out.print("Running Source Only. ");
            double[] performance = trainTestSimpleClassifier(upsampled_src_train, src_test, tgt_test);
            srcOnlyPerformance.add(new Performance(performance));
            double temp0 = performance[4];

            // Oracle- tgt only classifier
            System.out.print("Running Target Only. ");
            performance = trainTestSimpleClassifier(upsampled_tgt_train, src_test, tgt_test);
            tgtOnlyPerformance.add(new Performance(performance));

            // DiAd - radius
            System.out.print("Running DiAd radius. ");
            performance = diad_radius(upsampled_src_train, upsampled_tgt_train, src_test, tgt_test, max_iter_DiAd, radius);
            diadPerformance.add(new Performance(performance));
            double temp1 = performance[4];

            // DiAd - random
            System.out.print("Running DiAd random. ");
            performance = diad_random(upsampled_src_train, upsampled_tgt_train, src_test, tgt_test, max_iter_DiAd, numInstancesToSamplePerIter);
            diadRandomPerformance.add(new Performance(performance));
            double temp2 = performance[4];

            // S+T
            System.out.print("Running S+T. ");
            performance = sPlusT(upsampled_src_train, upsampled_tgt_train, src_test, tgt_test);
            sPlusTPerformance.add(new Performance(performance));

            // FE
//            System.out.print("Running FE Only. ");
//            performance = FE(upsampled_src_train, upsampled_tgt_train, src_test, tgt_test);
//            fePerformance.add(new Performance(performance));


            System.out.println("Finished Iter="+i+" target F1 of SourceOnly, Diad, DiAd random, and S+T="+temp0+", "+temp1 + ", "+temp2+" and "+performance[4]+"\n");

        }
        System.out.println("Printing final results");
        System.out.println("Source="+src_domain);
        System.out.println("Target="+tgt_domain);
        System.out.println("-------------------------------------");
        System.out.println("Source only experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(srcOnlyPerformance);
        System.out.println("-------------------------------------");
        System.out.println("Oracle (target only) experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(tgtOnlyPerformance);
        System.out.println("-------------------------------------");
        System.out.println("S+T experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(sPlusTPerformance);
        System.out.println("-------------------------------------");
        System.out.println("FE experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(fePerformance);
        System.out.println("-------------------------------------");
        System.out.println("DiAd random experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(diadRandomPerformance);
        System.out.println("-------------------------------------");
        System.out.println("DiAd experiment");
        System.out.println("-------------------------------------");
        Utils.printMedians(diadPerformance);
    }

    private static double[] trainTestSimpleClassifier(Instances train, Instances src_test, Instances tgt_test) throws Exception{
        System.out.println("Size of train set used ="+train.numInstances());

        // Train a model
        model.buildClassifier(train);

        //Test the classifier
        double[] ret = new double[6];

        // Testing on source test
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, src_test);
        ret[0] = eval.pctCorrect();
        ret[1] = eval.fMeasure(1)*100;
        ret[2] = eval.areaUnderROC(1)*100;

        // Testing on target test
        eval = new Evaluation(train);
        eval.evaluateModel(model, tgt_test);
        ret[3] = eval.pctCorrect();
        ret[4] = eval.fMeasure(1)*100;
        ret[5] = eval.areaUnderROC(1)*100;

        return ret;
    }

    private static double[] sPlusT(Instances upsampled_src_train, Instances upsampled_tgt_train, Instances src_test, Instances tgt_test) throws Exception{
        Instances surrogateLabelingTgtTrain = new Instances(upsampled_tgt_train); // create a copy so that original training data from target domain does not change

        model.buildClassifier(upsampled_src_train);

        // create training set for S+T
        Instances ST_train = new Instances(upsampled_src_train);
        for (int i = 0; i < surrogateLabelingTgtTrain.numInstances(); i++) {
            Instance ins = surrogateLabelingTgtTrain.instance(i);
            double pred = model.classifyInstance(ins);
            ins.setClassValue(pred);
            ST_train.add(ins);
        }

        return trainTestSimpleClassifier(ST_train, src_test, tgt_test);
    }

    private static double[] diad_radius(Instances upsampled_src_train, Instances upsampled_tgt_train, Instances src_test, Instances tgt_test, int num_max_iterations, double radius_threshold) throws Exception {
        Instances instance_train = new Instances(upsampled_src_train);
        Instances train_testSet = new Instances(upsampled_tgt_train);

        int ite_idx = 0;
        int cur_num_surr_inst = 0;
        double avg_numInstancesAddedEachIter = 0;
        int total_numberOfIterProcessed=0;

        for(ite_idx = 0; ite_idx < num_max_iterations; ite_idx++){
            Instances new_train = new Instances(instance_train, 0, 0);
//            new_train.remove(0);
            for(int k = 0; k < instance_train.numInstances(); k++){
                new_train.add(instance_train.instance(k));
            }

            model.buildClassifier(new_train);

            //Test the classifier
            Evaluation eTest = new Evaluation(train_testSet);
            eTest.evaluateModel(model, train_testSet);
            int n = train_testSet.numInstances();
//            ArrayList<Prediction>  ret_prediction = eTest.predictions();
            cur_num_surr_inst = 0;
            Instances new_train_testSet =  new Instances(train_testSet, 0, 0);
//            new_train_testSet.remove(0);//////////////////////
            //-----------------------------------------------------------------------------------------//
            //obtain surrogate instances
            //--------------------------------

            double[] distances = Utils.centroidDistanceTest(instance_train,train_testSet);
            for(int i = 0 ; i < n; i++){
                double cur_label = model.classifyInstance(train_testSet.instance(i));

                double[] confidences = model.distributionForInstance(train_testSet.instance(i));
                double cur_confidence = 0;
                if(confidences[0]>=0.5)
                    cur_confidence=confidences[0];
                else
                    cur_confidence=confidences[1];

                // Compute distance from (1,1)
                double cur_testDistance =  Math.hypot(1.0 - distances[i], 1.0 - cur_confidence);

//                    // Pick an instance if it lies in the circle
                if(cur_testDistance <= radius_threshold){
                    train_testSet.instance(i).setClassValue(cur_label);
                    instance_train.add(train_testSet.instance(i));
                    cur_num_surr_inst++;
                }
                else{
                    new_train_testSet.add(train_testSet.instance(i));
                }
            }

            train_testSet.delete();
            train_testSet = new Instances(new_train_testSet);
            new_train_testSet.delete();
            int cur_remaining_inst = train_testSet.numInstances();

            if(cur_num_surr_inst>0) {
                avg_numInstancesAddedEachIter += cur_num_surr_inst;
                total_numberOfIterProcessed++;
            }
//            System.out.println("current iteration: " + ite_idx);
//            System.out.println("size of set L ="+instance_train.size());
//            System.out.println("size of set U ="+train_testSet.size());
//            System.out.println("number of surrogate-inst: " + cur_num_surr_inst);
//            System.out.println("number of remaining test inst: " + cur_remaining_inst);

			/*determine whether the program should be terminated*/

////			 if no surrogate instances were added in this iteration, dump everything anyway to the training set with surrogate labels
//            model.buildClassifier(instance_train);
//            if(cur_num_surr_inst == 0 && cur_remaining_inst != 0) {
//                for(int x=0;x<train_testSet.numInstances();x++) {
//                    Instance ins = train_testSet.instance(x);
//                    double pred = model.classifyInstance(ins);
//                    ins.setClassValue(pred);
//                    instance_train.add(ins);
//                }
//                break;
//            }

            if(cur_num_surr_inst == 0 || cur_remaining_inst == 0){
                break;
            }
        }
//        System.out.println("Added an average of "+(avg_numInstancesAddedEachIter/total_numberOfIterProcessed) +" instances periter");

        return trainTestSimpleClassifier(instance_train, src_test,tgt_test);
    }

    private static double[] diad_random(Instances upsampled_src_train, Instances upsampled_tgt_train, Instances src_test, Instances tgt_test, int num_max_iterations, int numInstancesToAdd) throws Exception {
        Instances instance_train = new Instances(upsampled_src_train);
        Instances train_testSet = new Instances(upsampled_tgt_train);

        int ite_idx = 0;
        int cur_num_surr_inst = 0;

        for(ite_idx = 0; ite_idx < num_max_iterations; ite_idx++){

            // Train a classifier on currently labeled set
            Instances new_train = new Instances(instance_train, 0, 0);
//            new_train.remove(0);
            for(int k = 0; k < instance_train.numInstances(); k++){
                new_train.add(instance_train.instance(k));
            }

            model.buildClassifier(new_train);

            // Predict labels on unlabeled set using the  classifier
            Evaluation eTest = new Evaluation(train_testSet);
            eTest.evaluateModel(model, train_testSet);
            int n = train_testSet.numInstances();
//            ArrayList<Prediction>  ret_prediction = eTest.predictions();
            cur_num_surr_inst = 0;
            Instances new_train_testSet =  new Instances(train_testSet, 0, 0);
//            new_train_testSet.remove(0);

            // randomly sample instances (indices) to add to train set
            HashSet<Integer> indicesToAddToTrainingSet = sampler.randomSampler(n, numInstancesToAdd);

            // add randomly sampled instances to the train set
            for (int i = 0; i < n; i++) {
                double cur_label = model.classifyInstance(train_testSet.instance(i));

                if (indicesToAddToTrainingSet.contains(i)) {
                    train_testSet.instance(i).setClassValue(cur_label);
                    instance_train.add(train_testSet.instance(i));
                    cur_num_surr_inst++;
                } else {
                    new_train_testSet.add(train_testSet.instance(i));
                }
            }

            train_testSet.delete();
            train_testSet = new Instances(new_train_testSet);
            new_train_testSet.delete();
            int cur_remaining_inst = train_testSet.numInstances();

			/*determine whether the program should be terminated*/
            if(cur_num_surr_inst == 0 || cur_remaining_inst == 0){
                break;
            }
        }

        return trainTestSimpleClassifier(instance_train, src_test,tgt_test);
    }

    /**********This is FE Code. You might want to build on it **************/
//    private static double[] FE(Instances src_train, Instances tgt_train, Instances src_test, Instances tgt_test) throws Exception {
////        Instances src_train_copy = new Instances(src_train); // create a copy so that original training data from target domain does not change
//
//        // Surrogate labeling
//        model.buildClassifier(src_train);
//
//        Instances tgt_train_surrLabled = new Instances(tgt_train);
//        for (int i = 0; i < tgt_train_surrLabled.numInstances(); i++) {
//            Instance ins = tgt_train_surrLabled.instance(i);
//            double pred = model.classifyInstance(ins);
//            ins.setClassValue(pred);
//        }
//
//        // create feature vector for source
//        ArrayList<double[]> all = new ArrayList<>();
//        for(Instance src_inst:src_train) {
//            double[] attr = src_inst.toDoubleArray(); //second last is class
//
//            double label = attr[attr.length-2];
//            int D = attr.length-1;
//            double[] feats = new double[D];
//
//            for(int i=0, c=0;i<attr.length;i++)
//                if(i!=attr.length-2)
//                    feats[c++] = attr[i];
//
//            double newFeats[] = new double[D*3 +1];
//            for(int i=0;i<D;i++)
//                newFeats[i] = feats[i];
//            for(int i=D;i<2*D;i++)
//                newFeats[i] = feats[i-D];
//            for(int i=2*D;i<3*D;i++)
//                newFeats[i] =0;
//            newFeats[3*D] = label;
//
//            all.add(newFeats);
//
//            Utils.printArray(attr);
//            Utils.printArray(newFeats);
//        }
//          // next steps: write features to an arff file
    // do the same thing for target domain and write them to the same arff files
    // read data from arff files and train and test
//        return null;
//    }
//
//    public static void writeToArffFile(ArrayList<double[]> data, String arffFile){
//        try{
//
//            int numFeats = data.get(0).length - 1;
//
//            if(numFeats%3!=0){
//                System.err.println("Number of features to write is not divisible by 3!\t"+numFeats);
//                System.exit(-1);
//            }
//
//            PrintWriter writer = new PrintWriter(Paths.get(arffFile));
//            writer.println("@relation "+fileName);
//            for(int i=0; i< numFeats; i++){
//                writer.println("@attribute F_"+i+" numeric");
//            }
//            writer.println("@attribute label {0,1}");
//
//            writer.println("@data");
//            for(double[] fvec:data){
//                String label = fvec[numFeats];
//                //String label = ( fvec[numFeats] == 1 ) ? "POSITIVE":"NEGATIVE";
//                writer.println(StringUtils.join(fvec.subList(0, numFeats), ",")+","+ label);
//            }
//            writer.close();
//
//        }catch(Exception e) {
//            e.printStackTrace();
//        }
//    }


}

package edu.cs.cogcomp.Helpers;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import org.apache.commons.math3.distribution.BetaDistribution;

import java.util.HashSet;
import java.util.Random;

public class sampler {
    
    public static Random rm = new Random(26993);

	public static HashSet<Integer> randomSampler(int size, int k) {
		HashSet<Integer> samples = new HashSet<>();

		if(k>=size){ // if number of desired sampels>size of the data then no sampling needed
			for(int i=0;i<size;i++)
				samples.add(i);
			return samples;
		}

		while(samples.size()<k)
			samples.add(rm.nextInt(size));
		return samples;
	}

	public static HashSet<Integer> logDrawOptimized(double[] distribution, int k){
		HashSet<Integer> sampledIndices = new HashSet<>(); // takes care of sampling wihtout replacement
		if(k>=distribution.length){ // no sampling needed
			for(int i=0;i<distribution.length;i++)
				sampledIndices.add(i);
			return sampledIndices;
		}

		double normalizer = logNormalizer(distribution);

		//Draw Assignment
		while(sampledIndices.size()<k) {
			sampledIndices.add(sample(distribution, normalizer));
		}
		return sampledIndices;
	}

	public static int sample(double[] distribution, double normalizer){
		double r = Math.log(rm.nextDouble()) + normalizer;
		double sum = distribution[0];
		if (sum > r) return 0;
		for (int i = 1; i < distribution.length; i++) {
			sum = logSum(sum, distribution[i]);
			if (sum > r) return i;
		}
		return distribution.length - 1;
	}






	public static int logDrawOptimized(double[] distribution){
		double normalizer = logNormalizer(distribution);
		double r = Math.log(rm.nextDouble()) + normalizer;
		//Draw Assignment
		double sum = distribution[0];
		if (sum > r) return 0;
		for (int i=1 ; i<distribution.length ; i++){
			sum = logSum(sum,distribution[i]);
			if (sum > r) return i;
		}
		return distribution.length-1;
	}

	public static int logDraw(double[] distribution){
		double normalizer = logNormalizer(distribution);
		for (int i=0;i<distribution.length;i++) distribution[i] -= normalizer;
		//Draw Assignment
		return sample(distribution);
	}

	public static int sample(double[] distribution){
		double r = Math.log(rm.nextDouble());
		double sum = distribution[0];
		if (sum > r) return 0;
		for (int i=1 ; i<distribution.length ; i++){
			sum = logSum(sum,distribution[i]);
			if (sum > r) return i;
		}
		return 0;
}

    	public static double logNormalizer(double[] L){
		double T1 = L[0];
		double T2;
		for (int i=1;i<L.length;i++){
			T2 = L[i];
			T1 = logSum(T1,T2);
		}
		return T1;
	}


    public static int sample_cdf(double cdf[], int max_location){
        double rand = rm.nextDouble()*cdf[max_location-1];
        for(int i=0;i<max_location;i++){
            if(rand < cdf[i]){
                return i;
            }
        }
        return -1;
    }
    
    public static double sample_beta(double alpha, double beta){
        BetaDistribution b = new BetaDistribution(alpha,beta);
        return b.sample();
    }
    
	public static double computeSum(double[] distribution){
		double sum = 0;
		for(int i=0;i<distribution.length;i++) sum += distribution[i];
		if(sum<=0){
			System.err.println("Error: sum="+sum);
		}
		return sum;
	}
    
    public static int sample_multinomial(double[] distribution){
		double r = rm.nextDouble();
		double sum=0;
		for(int i=0;i<distribution.length;i++){
			sum += distribution[i];
			if(sum > r) return i;
		}
		return distribution.length-1;
    }
    
	public static int normDraw(double[] distribution){
		double sum = computeSum(distribution);
		double r = rm.nextDouble()*sum;
		sum=0;
		for(int i=0;i<distribution.length;i++){
			sum += distribution[i];
			if(sum > r) return i;
		}
		return distribution.length-1;
	}
    
	public static double logSum(double T1, double T2){
		double maximum,minimum;
		if (T1>T2){
	        maximum = T1;
	        minimum = T2;
		}
	    else{
	        maximum = T2;
	        minimum = T1;
	    }
	    return maximum + Math.log1p(Math.exp(minimum-maximum));
	}
	        
        public static double[] get_logCdf_from_logPdf(double[] L,int max_length){
		double T1 = L[0];
		double T2;
		for (int i=1;i<max_length;i++){
			T2 = L[i];
			T1 = logSum(T1,T2);
                        L[i] = T1;
		}
		return L;
	}
        
        public static int draw_log_cdf(double[] cdf, int max_lenght){
            double r= Math.log(rm.nextDouble()) + cdf[max_lenght-1];
            for(int i=0;i<max_lenght;i++){
                if(r < cdf[i]){
                    return i;
                }
            }
            System.err.println("Error");
            return -1;
        }


	public static boolean contains(int[] indicesToAddToTrainingSet, int x) {
		for(int i=0;i<indicesToAddToTrainingSet.length;i++)
			if(indicesToAddToTrainingSet[i] == x)
				return true;
		return false;
	}

	public static String print(HashSet<Integer> indicesToAddToTrainingSet) {
		String ret = "";
		for(Integer i:indicesToAddToTrainingSet)
			ret+=i+",";
		return ret;
	}

}

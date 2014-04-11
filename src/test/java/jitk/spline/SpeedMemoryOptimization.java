package jitk.spline;

import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class SpeedMemoryOptimization {

	double[][] srcPts;
	double[][] tgtPts;
	
	float[][] srcPtsF;
	float[][] tgtPtsF;
	
	int ndims;
	int N;     // # landmarks
	double ptScale = 10;
	double offScale = 1; // scale for offsets
	
	public static Random rand = new Random( 31415926536l );
	public Logger logger = LogManager.getLogger(SpeedMemoryOptimization.class.getName());
	
	public SpeedMemoryOptimization(int N, int ndims){
		setup(N,ndims,ptScale);
	}
	
	public SpeedMemoryOptimization(int N, int ndims, double scale){
		setup(N,ndims,scale);
	}
	
	public SpeedMemoryOptimization(int N, int ndims, float scale){
		setup(N,ndims,scale);
	}
	
	/**
	 * 
	 */
	public void setup(int N, int ndims, double scale){
		logger.info("setup double");
		this.N = N;
		this.ndims = ndims;
		this.ptScale = scale;
		srcPts = new double[ndims][N];
		tgtPts = new double[ndims][N];
		
		for (int d=0; d<ndims; d++) for( int i=0; i<N; i++ )
		{
			srcPts[d][i] = scale * rand.nextDouble();
			tgtPts[d][i] = srcPts[d][i] + offScale * rand.nextDouble();
			
		}
	}
	
	public void setup(int N, int ndims, float scale){
		logger.info("setup float");
		this.N = N;
		this.ndims = ndims;
		this.ptScale = scale;
		srcPtsF = new float[ndims][N];
		tgtPtsF = new float[ndims][N];
		
		for (int d=0; d<ndims; d++) for( int i=0; i<N; i++ )
		{
			srcPtsF[d][i] = (float)(scale * rand.nextDouble());
			tgtPtsF[d][i] = (float)(srcPtsF[d][i] + offScale * rand.nextDouble());
			
		}
	}
	
	/**
	 * Tried varying the linear system solver
	 * (in KernelTransform) for speed
	 */
	public void speed(){
	
		long startTime = System.currentTimeMillis();
//		
//		ThinPlateR2LogRSplineKernelTransformFloat tps = new ThinPlateR2LogRSplineKernelTransformFloat( ndims, srcPtsF, tgtPtsF );
//		tps.computeW();
//		
		long endTime = System.currentTimeMillis();
//		logger.info("(N="+N+") total time: " + (endTime-startTime) + "ms" );

		
		startTime = System.currentTimeMillis();
		
		ThinPlateR2LogRSplineKernelTransformFloatSep tpsSep = new ThinPlateR2LogRSplineKernelTransformFloatSep( ndims, srcPtsF, tgtPtsF );
		tpsSep.computeAffine();
		tpsSep.updateDisplacementPostAffine();
		tpsSep.computePostAffineDef();
		
		endTime = System.currentTimeMillis();
		logger.info("sep (N="+N+") total time: " + (endTime-startTime) + "ms" );
		
//		float[] pt = new float[ndims];
//		float[] tgt = new float[ndims];
//		
//		double avgDiffMag = 0;
//		double avgOrgErr  = 0;
//		double avgSepErr  = 0;
//		
//		for ( int i=0; i<N; i++){
//			
//			for (int d=0; d<ndims; d++){
//				pt[d]  = srcPtsF[d][i];
//				tgt[d] = tgtPtsF[d][i]; 
//			}
//			
//			float[] outOrg = tps.transformPoint(pt);
//			float[] outSep = tpsSep.transformPoint(pt);
//			
//			double diff = 0;
//			double orgErr = 0;
//			double sepErr = 0;
//			
//			for (int d=0; d<ndims; d++){
//				diff += (outOrg[d] - outSep[d]) * (outOrg[d] - outSep[d]);
//				orgErr += (outOrg[d] - tgt[d]) * (outOrg[d] - tgt[d]);
//				sepErr += (tgt[d] - outSep[d]) * (tgt[d] - outSep[d]);
//			}
//			diff = Math.sqrt(diff);
//			orgErr = Math.sqrt(orgErr);
//			sepErr = Math.sqrt(sepErr);
//			
//			avgDiffMag += diff;
//			avgOrgErr  += orgErr;
//			avgSepErr  += sepErr;
//		}
//		
//		avgDiffMag /= N;
//		
//		System.out.println("avgDiffMag: " + avgDiffMag);
//		System.out.println("avg Err Mag Orig: " + avgOrgErr);
//		System.out.println("avg Err Mag Sep : " + avgSepErr);
		
		
	}
	
	/**
	 * Tried varying the linear system solver
	 * (in KernelTransform) for speed
	 */
	public void memory(){
	
		long startTime = System.currentTimeMillis();
		
		ThinPlateR2LogRSplineKernelTransformFloat tps = new ThinPlateR2LogRSplineKernelTransformFloat( ndims, srcPtsF, tgtPtsF );
		tps.computeW();
		
		long endTime = System.currentTimeMillis();
		logger.info("(N="+N+") total time: " + (endTime-startTime) + "ms" );
		
	}
	
	public static void main(String[] args) {
		
		System.out.println("starting");
		
//		int[] NList = new int[]{20, 50, 100};
		int[] NList = new int[]{2000};
		
//		int ndims = 3;
		int ndims = 2;
		
//		int numTrials = 15;
		int numTrials = 2;
		
		for (int i=0; i<NList.length; i++)
		{
			SpeedMemoryOptimization smo = new SpeedMemoryOptimization( NList[i], ndims, 10f );
			for( int t=0; t<numTrials; t++){
				smo.speed();
			}
		}
		
		
		System.out.println("finished");
		System.exit(0);
	}

}

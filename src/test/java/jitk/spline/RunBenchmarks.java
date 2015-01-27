package jitk.spline;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.janelia.utility.benchmark.Benchmark;
import org.janelia.utility.benchmark.ExecutionFunctor;

import com.google.common.base.Strings;

public class RunBenchmarks {
	
	protected static Logger logger = LogManager.getLogger(RunBenchmarks.class.getName());
	
	public final double[][] srcPts;
	public final double[][] tgtPts;
	
	public final float[][] srcPts_F;
	public final float[][] tgtPts_F;
	
	final int nLandmarks = 200; 
	final int ndims      =   3;
	
	double multiplier = 20;
	double testmultiplier = 20;
	
	float multiplier_F = 20;
	float testmultiplier_F = 20;
	
	int Nwarmup =  20;
	int Nrep    = 100;
	int Niters  = 200;
	
	int NtestPts = 1000;
	
	Random rng;
	
	public RunBenchmarks() {
		
		rng = new Random();
		
		srcPts = new double[ndims][nLandmarks];
		tgtPts = new double[ndims][nLandmarks];
		
		for (int d = 0; d < ndims; d++) {
			for (int i = 0; i < nLandmarks; i++) {
				srcPts[d][i] = multiplier * rng.nextDouble();
				tgtPts[d][i] = srcPts[d][i] + rng.nextDouble();
			}
		}
		
		srcPts_F = new float[ndims][nLandmarks];
		tgtPts_F = new float[ndims][nLandmarks];
		
		for (int d = 0; d < ndims; d++) {
			for (int i = 0; i < nLandmarks; i++) {
				srcPts_F[d][i] = multiplier_F * rng.nextFloat();
				tgtPts_F[d][i] = srcPts_F[d][i] + rng.nextFloat();
			}
		}
	}

	public void runSolveBenchmark( KernelTransform.SolverType type) {
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		
		tps.solverType = type;
		logger.info("Using solver " + tps.solverType );
	
		BaselineRunner runner = this.new BaselineRunner( tps );
		
		Benchmark benchmark = new Benchmark( runner );
		benchmark.evaluate( Nwarmup, Nrep, Niters );
		
		logger.info("runtime: " + benchmark.getMean() + "(" + benchmark.getVar() + ")");
	}
	
	public void runFloatVsDoubleAccuracy(){
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts_F, tgtPts_F );
		tps.solverType = KernelTransform.SolverType.LINEAR;
		
		
		ThinPlateR2LogRSplineKernelTransformFloat tps_F = new ThinPlateR2LogRSplineKernelTransformFloat( 
				ndims, srcPts_F, tgtPts_F );
		
		tps.solve();
		tps_F.solve();
		
		float[] pt = new float[ndims];
		double[] pairDistList = new double[NtestPts];
		
		for( int i=0; i < NtestPts; i++ ) {
			
			// init the point to transform
			for(int d=0; d < ndims; d++ ) {
				pt[d] = testmultiplier_F * rng.nextFloat();
			}
			
			float[] ptXfm = toFloat(tps.transform( toDouble( pt ) ));
			float[] ptXfm_F =  tps_F.transformPoint(pt);
			
//			pairDistList[i] = norm(toDouble(XfmUtils.subtract( ptXfm, ptXfm_F)));
			pairDistList[i] = norm(toDouble(XfmUtils.subtract( ptXfm, ptXfm_F)));
		}
		
		double mnDiff = new Mean().evaluate(pairDistList);
		System.out.println(
				" mean dist for \t"
				+ "LINEAR-FLOAT "
				+ " vs "
				+ "LINEAR-DOUBLE"
				+ " : "
				+ mnDiff);
		
	}
	
	public double[] toDouble( float[] in ){
		double[] out = new double[ in.length ];
		for( int i=0; i<in.length; i++ )
			out[i] = in[i];
		
		return out;
	}
	
	public float[] toFloat( double[] in ){
		float[] out = new float[ in.length ];
		for( int i=0; i<in.length; i++ )
			out[i] = (float)in[i];
		
		return out;
	}
	
	public void runSolverConsistency(){
		
		ThinPlateR2LogRSplineKernelTransform tpsP = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		tpsP.solverType = KernelTransform.SolverType.PSEUDOINVERSE;
		
		ThinPlateR2LogRSplineKernelTransform tpsL = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		tpsL.solverType = KernelTransform.SolverType.LINEAR;
		
		ThinPlateR2LogRSplineKernelTransform tpsG = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		tpsG.solverType = KernelTransform.SolverType.GENERAL;
		
		ThinPlateR2LogRSplineKernelTransform tpsS = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		tpsS.solverType = KernelTransform.SolverType.LEAST_SQUARES;
		
		ThinPlateR2LogRSplineKernelTransform tpsQ = new ThinPlateR2LogRSplineKernelTransform( 
				ndims, srcPts, tgtPts );
		tpsQ.solverType = KernelTransform.SolverType.LEAST_SQUARES_QR;
		
		ArrayList<KernelTransform> tpsList = new ArrayList<KernelTransform>();
		
		tpsList.add( tpsP );
		tpsList.add( tpsL );
		tpsList.add( tpsG );
		tpsList.add( tpsS );
		tpsList.add( tpsQ );
		
		// solve all
		for( KernelTransform tps : tpsList ){
			tps.solve();
		}
		int M = tpsList.size();
		
		double[] pt = new double[ndims];
		ArrayList<double[]> xfmPts = new ArrayList<double[]>();
		ArrayList<double[]> pairDistList = new ArrayList<double[]>();
		
		
		for( int i=0; i < NtestPts; i++ ) {
			
			// init the point to transform
			for(int d=0; d < ndims; d++ ) {
				pt[d] = testmultiplier * rng.nextDouble();
			}
			
			xfmPts.clear();
			for( int j=0; j < M; j++ ) {
				xfmPts.add( tpsList.get(j).transform(pt) );
			}
			
			pairDistList.add( pairwiseDists( xfmPts ));
		}
	
		
		
		int Mchoose2 = pairDistList.get(0).length;
		System.out.println( pairDistList.size() + " x " + Mchoose2 );
			
		
		int k = 0;
		for (int i = 0; i < M; i++) {
			for (int j = i + 1; j < M; j++) {

				System.out.println(
						" mean dist for \t"
						+ Strings.padEnd( tpsList.get(i).solverType.toString(), 16, ' ' ) 
						+ " vs "
						+ Strings.padEnd( tpsList.get(j).solverType.toString(), 16, ' ' ) 
						+ " : "
						+ meanAcrossColumn(pairDistList, k));
				k++;
			}
		}
	}
	
	public static double meanAcrossColumn( ArrayList<double[]> mtx, int col ){
		double mean = 0;
		for( double[] row : mtx ){
			mean += row[col];
		}
		mean /= mtx.size();
		return mean;
	}

	public static double[] pairwiseDists(ArrayList<double[]> ptList) {
		int N = ptList.size();
		double[] out = new double[N * (N - 1)];
		int k = 0;
		for (int i = 0; i < N; i++) {
			for (int j = i + 1; j < N; j++) {
				double[] diff = XfmUtils.subtract(ptList.get(i), ptList.get(j));
				out[k++] = norm(diff);
			}
		}
		return out;
	}

	public static double norm(final double[] v) {
		double nrm = 0;
		for (int i = 0; i < v.length; i++) {
			nrm += v[i] * v[i];
		}
		nrm = Math.sqrt(nrm);
		return nrm;
	}
	
	public static void main( String[] args ) {
		//logger.info("Starting");
		
		RunBenchmarks rb = new RunBenchmarks();
//		rb.runSolveBenchmark(KernelTransform.SolverType.PSEUDOINVERSE);
//		rb.runSolverConsistency();
		rb.runFloatVsDoubleAccuracy();
		
		//logger.info("Finished");
	}
	
	public class BaselineRunner implements ExecutionFunctor {

		KernelTransform xfm;
		
		public BaselineRunner(KernelTransform xfm){
			this.xfm = xfm;
		}
		
		@Override
		public void run() {
			xfm.solve();
		}
		
	}

}

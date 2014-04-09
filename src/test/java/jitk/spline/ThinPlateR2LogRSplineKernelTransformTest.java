package jitk.spline;

import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.junit.Test;

import static org.junit.Assert.*;


public class ThinPlateR2LogRSplineKernelTransformTest {
	
	public static double tol = 0.0001;
	public static Random rand = new Random( 31415926536l );
	
	double[][] srcPts;
	double[][] tgtPts;
	
	float[][] srcPtsF;
	float[][] tgtPtsF;
	
	int ndims;
	int N;
	
	public Logger logger = LogManager.getLogger(ThinPlateR2LogRSplineKernelTransformTest.class.getName());
	
	
	public void genPtListNoAffine1(){
		N = 50;
		int D = 10;
		ndims = 3;
		
		srcPts = new double[3][2*N];
		tgtPts = new double[3][2*N];
		
		int k = 0;
		for( int i=0; i<N; i++ )
		{
			
			double[] off = new double[]{
					rand.nextDouble(),
					rand.nextDouble(),
					rand.nextDouble() };
			
			srcPts[0][k] = D*rand.nextDouble();
			srcPts[1][k] = D*rand.nextDouble();
			srcPts[2][k] = D*rand.nextDouble();
			
			tgtPts[0][k] = srcPts[0][k] + off[0];
			tgtPts[1][k] = srcPts[1][k] + off[1];
			tgtPts[2][k] = srcPts[2][k] + off[2];
			k++;
			
			srcPts[0][k] = -srcPts[0][k-1];
			srcPts[1][k] = -srcPts[1][k-1];
			srcPts[2][k] = -srcPts[2][k-1];
			
			tgtPts[0][k] = srcPts[0][k] - off[0];
			tgtPts[1][k] = srcPts[1][k] - off[1];
			tgtPts[2][k] = srcPts[2][k] - off[2];
			k++;
			
		}
	}
	
	@Test
	public void testAffineOnly(){
		ndims = 3;
		srcPts = new double[][]
				{
				{0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};
		
		
		double[][] aff = new double[][]{
				{0, 1, 0},
				{0, 0, 1},
				{1, 0, 0} };
		
		N = srcPts[0].length;
		tgtPts = XfmUtils.genPtListAffine(srcPts, aff);
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		tps.setDoAffine(false);
		tps.computeW();
		
		double[] testPt = new double[ndims];
		for( int n=0; n<N; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = srcPts[i][n];
			}
			
			double[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("pure affine transformation", tgtPts[i][n], outPt[i], tol);
			}
		}
	}

	@Test 
	public void testIdentitySmall2d(){
		int ndims = 2;
		
		float[][] pts = new float[][]
				{
				{-1,0,1}, // x
				{-1,0,1}  // y
				};
		
//		double[][] ptsd = new double[][]
//				{
//				{-1,0,1}, // x
//				{-1,0,1}  // y
//				};
		
		int nL = pts[0].length;
		
		float[][] tpts   = XfmUtils.genPtListScale(pts , new double[]{2,3});
		
		ThinPlateR2LogRSplineKernelTransformFloat tps = new ThinPlateR2LogRSplineKernelTransformFloat( ndims, pts, tpts );
		tps.computeW();
		
		
		float[] testPt = new float[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			float[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tpts[i][n], outPt[i], tol);
			}
		}
	}
	
	@Test 
	public void testIdentitySmall3d(){
		int ndims = 3;
		
		float[][] pts = new float[][]
				{
				{0,0,0,0,0,0,1,1}, // x
				{0,0,0,1,1,1,2,2}, // y
				{0,1,2,0,1,2,0,1}  // z					
				};
		
//		double[][] ptsd = new double[][]
//				{
//				{-1,0,1},  // x
//				{-1,0,1},  // y
//				{-1,0,1}   // z
//				};
		
		int nL = pts[0].length;
		
		float[][] tpts   = XfmUtils.genPtListScale(pts , new double[]{2,3,0.5});
		
		ThinPlateR2LogRSplineKernelTransformFloat tps = new ThinPlateR2LogRSplineKernelTransformFloat( ndims, pts, tpts );
		tps.computeW();
		
		
		float[] testPt = new float[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			float[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tpts[i][n], outPt[i], tol);
			}
		}
	}
	
	@Test 
	public void testIdentity(){
		int ndims = 3;
		
		double[][] pts = new double[][]
				{
				{-1,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{-1,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{-1,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};
		int nL = pts[0].length;
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tps.computeW();
		
		double[] testPt = new double[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			double[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}
		
		ThinPlateR2LogRSplineKernelTransform tpsNA = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tpsNA.setDoAffine(false);
		tpsNA.computeW();
		
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			double[] outPt = tpsNA.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}
		
	}
	
	@Test
	public void testScale3d() {

		int ndims = 3;
		double[][] src_simple = new double[][]
				{
				{0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};

		double[][] tgtPtList = XfmUtils.genPtListScale(src_simple, new double[]{2, 0.5, 4});

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgtPtList );
		tps.computeW();


		double[] srcPt = new double[]{ 0.0f, 0.0f, 0.0f};
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);
		assertEquals("scale z1", 0, ptXfm[2], tol);

		srcPt = new double[]{ 0.5f, 0.5f, 0.5f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);
		assertEquals("scale z2", 2.00, ptXfm[2], tol);

		srcPt = new double[]{ 1.0f, 1.0f, 1.0f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);
		assertEquals("scale z3", 4.0, ptXfm[2], tol);
	}

	@Test 
	public void testAffineReasonable(){
		
		genPtListNoAffine1();
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		//tps.setDoAffine(false);
		tps.computeW();
		
		double[][] a = tps.getAffine();
		double[] t   = tps.getTranslation();
		
		System.out.println("a: " + XfmUtils.printArray(a)+"\n");
		System.out.println("t: " + XfmUtils.printArray(t));
		
		double[] testPt = new double[ndims];
		for( int n=0; n<2*N; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = srcPts[i][n];
			}
			
			double[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tgtPts[i][n], outPt[i], tol);
			}
		}
	}
	
	@Test
	public void testAffineSanity() {
		
		int ndims = 2;
		double[][] src = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		
		double[][] tgt = new double[src.length][src[0].length];
		
		for(int i=0; i<src.length; i++)for(int j=0; j<src[0].length; j++){
			tgt[i][j] = src[i][j] + Math.random()*0.1;
		}
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src, tgt );
		tps.setDoAffine(false);
		tps.computeW();
		
		//System.out.println(" aMatrix: (Expect all zeros)\n" + 
		//		printArray(tps.aMatrix) + "\n");
		assertTrue("aMatrix should be null", tps.aMatrix==null);
		
		//System.out.println(" bVector: (Expect all zeros)\n" + 
		//		printArray(tps.bVector) + "\n");
		assertTrue("bVector should be null", tps.bVector==null);


		//System.out.println(" dMatrix: (Expect non-zero)\n" + tps.dMatrix + "\n");
		boolean isNonZeroMtxElem = false;
      for(int i=0; i<tps.dMatrix.getNumElements(); i++)
      {
         isNonZeroMtxElem = isNonZeroMtxElem || ( tps.dMatrix.get(i) != 0 );
      }
      assertTrue(" dMatrix as non-zero element", isNonZeroMtxElem );
		
	}
	

	
	@Test
	public void testScale2() {
		ndims = 2;
		N     = 6;
		srcPtsF = new float[ndims][N];
		for(int d=0; d<ndims; d++)for( int i=0; i<N; i++){
			srcPtsF[d][i] = (float)( 10 * rand.nextDouble());
		}
		tgtPtsF = XfmUtils.genPtListScale(srcPtsF, new double[]{2, 0.5});
		
		ThinPlateR2LogRSplineKernelTransformFloat tps = new ThinPlateR2LogRSplineKernelTransformFloat( ndims, srcPtsF, tgtPtsF );
		tps.computeW();
		
		float[] testPt = new float[ndims];
		for( int n=0; n<N; n++) {
			
			for( int i=0; i<ndims; i++) {
				testPt[i] = srcPtsF[i][n];
			}
			
			float[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tgtPtsF[i][n], outPt[i], tol);
			}
		}
		
	}
	
	@Test
	public void testTransXfmAffineTps(){
		
//		int ndims = 2;
//		float[][] src_simple = new float[][]
//				{
//				{-1,-1,-1,1,1,1,2,2,2}, // x
//				{-1,1,2,-1,1,2,0,1,2}, // y
//				};
//		
//		// target points
//		float[][] tgt= new float[][]
//				{
//				{ -0.5f, -0.5f, -0.5f, 1.5f, 1.5f, 1.5f, 2.0f, 2.0f, 2.0f}, // x
//				{ -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f } // y
//				};
		
		int ndims = 2;
		srcPtsF = new float[][]
				{
				{-1,-1,-1,1,1,1,2,2,2}, // x
				{-1,1,2,-1,1,2,-1,1,2}, // y
				};
		
		// target points
		tgtPtsF= new float[][]
				{
					{ 0,0,0, 2,2,2, 3,3,3}, // x
					{ 1,3,4, 1,3,4, 1,3,5 } // y
				};

		ThinPlateR2LogRSplineKernelTransformFloat tps 
			= new ThinPlateR2LogRSplineKernelTransformFloat( ndims, srcPtsF, tgtPtsF);

		tps.computeAffine();
		tps.updateDisplacementPostAffine();
		
//		N = srcPtsF[0].length;
//		float[] testPt = new float[ndims];
//		for( int n=0; n<N; n++) {
//
//			for( int d=0; d<ndims; d++) {
//				testPt[d] = srcPtsF[d][n];
//			}
//
//			float[] outPt = tps.transformPointAffine(testPt);
//			for( int d=0; d<ndims; d++) {
//				assertEquals("translation, use affine", tgtPtsF[d][n], outPt[d], tol);
//			}
//		}
	
		tps.computePostAffineDef();
		
	}
	
	@Test
	public void testScale() {

		int ndims = 2;
		srcPts = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};

		tgtPts = XfmUtils.genPtListScale(srcPts, new double[]{2, 0.5});
		
		logger.debug("srcPts:\n" + XfmUtils.printArray(srcPts));
		logger.debug("\ntgtPts:\n" + XfmUtils.printArray(tgtPts));

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		tps.computeW();


		double[] srcPt = new double[]{ 0.0f, 0.0f };
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);

		srcPt = new double[]{ 0.5f, 0.5f };	
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);

		srcPt = new double[]{ 1.0f, 1.0f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);
		
		
		N = srcPts[0].length;
		double[] testPt = new double[ndims];
		for( int n=0; n<N; n++) {
			
			for( int d=0; d<ndims; d++) {
				testPt[d] = srcPts[d][n];
			}
			
			double[] outPt = tps.transformPoint(testPt);
			for( int d=0; d<ndims; d++) {
				assertEquals("Identity transformation", tgtPts[d][n], outPt[d], tol);
			}
		}
	}

	@Test
	public void testWarp() {

		int ndims = 2;
		double[][] src_simple = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		// target points
		double[][] tgt= new double[][]
				{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0}, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
				};

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgt);
		tps.computeW();
		//tps.printLandmarks();

		double[] srcPt = new double[]{0.0f,0.0f};
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("warp x1", -0.5, ptXfm[0], tol);
		assertEquals("warp y1", -0.5, ptXfm[1], tol);

		srcPt = new double[]{0.5f,0.5f};
		ptXfm = tps.transformPoint(srcPt);
		
		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals("warp x2", 0.6241617, ptXfm[0], tol);
		assertEquals("warp y2", 0.6241617, ptXfm[1], tol);

		srcPt = new double[]{1.0f,1.0f};
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("warp x3", 1.5, ptXfm[0], tol);
		assertEquals("warp y3", 1.5, ptXfm[1], tol);

	}

}

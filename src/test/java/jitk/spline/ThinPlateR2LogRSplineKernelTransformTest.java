package jitk.spline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.junit.Test;


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
		final int D = 10;
		ndims = 3;

		srcPts = new double[3][2*N];
		tgtPts = new double[3][2*N];

		int k = 0;
		for( int i=0; i<N; i++ )
		{

			final double[] off = new double[]{
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


		final double[][] aff = new double[][]{
				{0, 1, 0},
				{0, 0, 1},
				{1, 0, 0} };

		N = srcPts[0].length;
		tgtPts = XfmUtils.genPtListAffine(srcPts, aff);

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		tps.setDoAffine(false);
		tps.solve();

		final double[] testPt = new double[ndims];
		for( int n=0; n<N; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = srcPts[i][n];
			}

			final double[] outPt = tps.apply(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("pure affine transformation", tgtPts[i][n], outPt[i], tol);
			}
		}
	}

	@Test
	public void testIdentitySmall2d(){
		final int ndims = 2;

		final double[][] pts = new double[][]
				{
				{-1,0,1}, // x
				{-1,0,1}  // y
				};

		final int nL = pts[0].length;

		final double[][] tpts   = XfmUtils.genPtListScale(pts , new double[]{2,3});

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, tpts );
		tps.solve();


		final double[] testPt = new double[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}

			final double[] outPt = tps.apply(testPt);
			System.out.println( outPt.length );
			System.out.println( tpts.length + " x " + tpts[0].length );
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tpts[i][n], outPt[i], tol);
			}
		}
	}

	@Test
	public void testIdentitySmall3d(){
		final int ndims = 3;


		final double[][] pts = new double[][]
				{
				{0,0,0,0,0,0,1,1}, // x
				{0,0,0,1,1,1,2,2}, // y
				{0,1,2,0,1,2,0,1}  // z
				};

		final int nL = pts[0].length;

		final double[][] tpts   = XfmUtils.genPtListScale(pts , new double[]{2,3,0.5});

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, tpts );
		tps.solve();


		final double[] testPt = new double[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}

			final double[] outPt = tps.apply(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tpts[i][n], outPt[i], tol);
			}
		}
	}

	@Test
	public void testIdentity(){
		final int ndims = 3;

		final double[][] pts = new double[][]
				{
				{-1,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{-1,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{-1,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};
		final int nL = pts[0].length;

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tps.solve();

		final double[] testPt = new double[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}

			final double[] outPt = tps.apply(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}

		final ThinPlateR2LogRSplineKernelTransform tpsNA = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tpsNA.setDoAffine(false);
		tpsNA.solve();

		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}

			final double[] outPt = tpsNA.apply(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}

	}

	@Test
	public void testScale3d() {

		final int ndims = 3;
		final double[][] src_simple = new double[][]
				{
				{0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};

		final double[][] tgtPtList = XfmUtils.genPtListScale(src_simple, new double[]{2, 0.5, 4});

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgtPtList );
		tps.solve();


		double[] srcPt = new double[]{ 0.0f, 0.0f, 0.0f};
		double[] ptXfm = tps.apply(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);
		assertEquals("scale z1", 0, ptXfm[2], tol);

		srcPt = new double[]{ 0.5f, 0.5f, 0.5f };
		ptXfm = tps.apply(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);
		assertEquals("scale z2", 2.00, ptXfm[2], tol);

		srcPt = new double[]{ 1.0f, 1.0f, 1.0f };
		ptXfm = tps.apply(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);
		assertEquals("scale z3", 4.0, ptXfm[2], tol);
	}

	@Test
	public void testAffineReasonable(){

		genPtListNoAffine1();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		//tps.setDoAffine(false);
		tps.solve();

		final double[][] a = tps.getAffine();
		final double[] t   = tps.getTranslation();

		System.out.println("a: " + XfmUtils.printArray(a)+"\n");
		System.out.println("t: " + XfmUtils.printArray(t));

		final double[] testPt = new double[ndims];
		for( int n=0; n<2*N; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = srcPts[i][n];
			}

			final double[] outPt = tps.apply(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", tgtPts[i][n], outPt[i], tol);
			}
		}
	}

	@Test
	public void testAffineSanity() {

		final int ndims = 2;
		final double[][] src = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};

		final double[][] tgt = new double[src.length][src[0].length];

		for(int i=0; i<src.length; i++)for(int j=0; j<src[0].length; j++){
			tgt[i][j] = src[i][j] + Math.random()*0.1;
		}

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src, tgt );
		tps.setDoAffine(false);
		tps.solve();

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

//	@Test
//	public void testTransXfmAffineTps(){
//
////		int ndims = 2;
////		float[][] src_simple = new float[][]
////				{
////				{-1,-1,-1,1,1,1,2,2,2}, // x
////				{-1,1,2,-1,1,2,0,1,2}, // y
////				};
////
////		// target points
////		float[][] tgt= new float[][]
////				{
////				{ -0.5f, -0.5f, -0.5f, 1.5f, 1.5f, 1.5f, 2.0f, 2.0f, 2.0f}, // x
////				{ -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f } // y
////				};
//
//		int ndims = 2;
//		srcPtsF = new float[][]
//				{
//				{-1,-1,-1,1,1,1,2,2,2}, // x
//				{-1,1,2,-1,1,2,-1,1,2}, // y
//				};
//
//		// target points
//		tgtPtsF= new float[][]
//				{
//					{ 0,0,0, 2,2,2, 3,3,3}, // x
//					{ 1,3,4, 1,3,4, 1,3,5 } // y
//				};
//
//		ThinPlateR2LogRSplineKernelTransformFloatSep tps
//			= new ThinPlateR2LogRSplineKernelTransformFloatSep( ndims, srcPtsF, tgtPtsF);
//
//		tps.fit();
//
//		N = srcPtsF[0].length;
//		float[] testPt = new float[ndims];
//		for( int n=0; n<N; n++) {
//
//			for( int d=0; d<ndims; d++) {
//				testPt[d] = srcPtsF[d][n];
//			}
//
//			float[] outPt = tps.transform(testPt);
//			logger.debug("point: " + XfmUtils.printArray(testPt) + " -> " + XfmUtils.printArray(outPt));
//			for( int d=0; d<ndims; d++) {
//				assertEquals("translation, use affine", tgtPtsF[d][n], outPt[d], tol);
//			}
//		}
//
//	}

	@Test
	public void testScale() {

		final int ndims = 2;
		srcPts = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};

		tgtPts = XfmUtils.genPtListScale(srcPts, new double[]{2, 0.5});

		logger.debug("srcPts:\n" + XfmUtils.printArray(srcPts));
		logger.debug("\ntgtPts:\n" + XfmUtils.printArray(tgtPts));

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, srcPts, tgtPts );
		tps.solve();


		double[] srcPt = new double[]{ 0.0f, 0.0f };
		double[] ptXfm = tps.apply(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);

		srcPt = new double[]{ 0.5f, 0.5f };
		ptXfm = tps.apply(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);

		srcPt = new double[]{ 1.0f, 1.0f };
		ptXfm = tps.apply(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);


		N = srcPts[0].length;
		final double[] testPt = new double[ndims];
		for( int n=0; n<N; n++) {

			for( int d=0; d<ndims; d++) {
				testPt[d] = srcPts[d][n];
			}

			final double[] outPt = tps.apply(testPt);
			for( int d=0; d<ndims; d++) {
				assertEquals("Identity transformation", tgtPts[d][n], outPt[d], tol);
			}
		}
	}

	@Test
	public void testWarp() {

		final int ndims = 2;
		final double[][] src_simple = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		// target points
		final double[][] tgt= new double[][]
				{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0}, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
				};

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgt);
		tps.solve();
		//tps.printLandmarks();

		/* **** PT 2 **** */
		double[] srcPt = new double[]{0.0f,0.0f};
		double[] ptXfm = tps.apply(srcPt);
		assertEquals("warp x1", -0.5, ptXfm[0], tol);
		assertEquals("warp y1", -0.5, ptXfm[1], tol);

		// check double
		final double[] srcPtF = srcPt.clone();
		final double[] ptXfmF = tps.apply(srcPtF);
		assertEquals("warp x1 float", -0.5f, ptXfmF[0], tol);
		assertEquals("warp y1 float", -0.5f, ptXfmF[1], tol);

		// double in place
		tps.applyInPlace(srcPt);
		assertEquals("warp x1 in place", -0.5, srcPt[0], tol);
		assertEquals("warp y1 in place", -0.5, srcPt[1], tol);

		/* **** PT 2 **** */
		srcPt = new double[]{0.5f,0.5f};
		ptXfm = tps.apply(srcPt);

		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals("warp x2", 0.6241617, ptXfm[0], tol);
		assertEquals("warp y2", 0.6241617, ptXfm[1], tol);

		// double in place 2
		tps.applyInPlace(srcPt);
		assertEquals("warp x2 in place", 0.6241617, srcPt[0], tol);
		assertEquals("warp y2 in place", 0.6241617, srcPt[1], tol);

		/* **** PT 3 **** */
		srcPt = new double[]{1.0f,1.0f};
		ptXfm = tps.apply(srcPt);
		assertEquals("warp x3", 1.5, ptXfm[0], tol);
		assertEquals("warp y3", 1.5, ptXfm[1], tol);

		tps.applyInPlace(srcPt);
		assertEquals("warp x3 in place", 1.5, srcPt[0], tol);
		assertEquals("warp y3 in place", 1.5, srcPt[1], tol);
	}

	@Test
	public void testAddPoints() {

		final int ndims = 2;

		final ThinPlateR2LogRSplineKernelTransform tps
			= new ThinPlateR2LogRSplineKernelTransform( ndims );

		for( int i = 0; i<200; i++){
			tps.addMatch( new double[]{ Math.random(), Math.random() },
					 	  new double[]{ Math.random(), Math.random() } );
		}

	}

	@Test
	public void testWeights() {

		final int ndims = 2;
		final double[][] src_simple = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		// target points
		final double[][] tgt= new double[][]
				{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0}, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
				};
//		double[] weights = new double[]
//				{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

		final double[] weights = new double[]
				{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0};

		final ThinPlateR2LogRSplineKernelTransform tps
			= new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgt, weights);
		tps.solve();
		//tps.printLandmarks();

		double[] srcPt = new double[]{0.0f,0.0f};
		double[] ptXfm = tps.apply(srcPt);
		assertEquals("warp x1", -0.5, ptXfm[0], tol);
		assertEquals("warp y1", -0.5, ptXfm[1], tol);

		srcPt = new double[]{0.5f,0.5f};
		ptXfm = tps.apply(srcPt);

		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals("warp x2", 0.6241617, ptXfm[0], tol);
		assertEquals("warp y2", 0.6241617, ptXfm[1], tol);

		srcPt = new double[]{1.0f,1.0f};
		ptXfm = tps.apply(srcPt);
		assertEquals("warp x3", 1.5, ptXfm[0], tol);
		assertEquals("warp y3", 1.5, ptXfm[1], tol);

	}

}

package jitk.spline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.junit.Test;

public class ThinPlateR2LogRSplineKernelTransformTest
{

	public static double tol = 0.0001;
	public static Random rand = new Random( 31415926536l );

	double[][] srcPts;
	double[][] tgtPts;

	float[][] srcPtsF;
	float[][] tgtPtsF;

	int ndims;
	int N;

	public Logger logger = LogManager
			.getLogger( ThinPlateR2LogRSplineKernelTransformTest.class.getName() );

	public void genPtListSimple2d()
	{
		ndims = 2;
		srcPts = new double[][]{
				{ -1.0, 0.0, 1.0, 0.0 }, // x
				{ 0.0, -1.0, 0.0, 1.0 } }; // y

		tgtPts = new double[][]{
				{ -2.0, 0.0, 2.0, 0.0 }, // x
				{ 0.0, -2.0, 0.0, 2.0 } }; // y

	}

	public void genPtListNoAffine1()
	{
		genPtListNoAffine1( 50, 10 );
	}

	public void genPtListNoAffine1( int N, int D )
	{
		ndims = 3;
		srcPts = new double[ 3 ][ 2 * N ];
		tgtPts = new double[ 3 ][ 2 * N ];

		int k = 0;
		for ( int i = 0; i < N; i++ )
		{

			final double[] off = new double[]
			{ rand.nextDouble(), rand.nextDouble(), rand.nextDouble() };

			srcPts[ 0 ][ k ] = D * rand.nextDouble();
			srcPts[ 1 ][ k ] = D * rand.nextDouble();
			srcPts[ 2 ][ k ] = D * rand.nextDouble();

			tgtPts[ 0 ][ k ] = srcPts[ 0 ][ k ] + off[ 0 ];
			tgtPts[ 1 ][ k ] = srcPts[ 1 ][ k ] + off[ 1 ];
			tgtPts[ 2 ][ k ] = srcPts[ 2 ][ k ] + off[ 2 ];
			k++;

			srcPts[ 0 ][ k ] = -srcPts[ 0 ][ k - 1 ];
			srcPts[ 1 ][ k ] = -srcPts[ 1 ][ k - 1 ];
			srcPts[ 2 ][ k ] = -srcPts[ 2 ][ k - 1 ];

			tgtPts[ 0 ][ k ] = srcPts[ 0 ][ k ] - off[ 0 ];
			tgtPts[ 1 ][ k ] = srcPts[ 1 ][ k ] - off[ 1 ];
			tgtPts[ 2 ][ k ] = srcPts[ 2 ][ k ] - off[ 2 ];
			k++;

		}
	}

	@Test
	public void testTPSInverseConvenience()
	{
		final double[] target = new double[]{ 0.5, 0.5 };
		final double tolerance = 0.01;
		final int maxIters = 9999;

		genPtListSimple2d();
		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts, true );

		double[] invResult = new double[ 2 ];

		double finalError = tps.inverse( target, invResult, tolerance, maxIters );

		double[] invResultXfm = tps.apply( invResult );
		logger.debug( "final error   : " + finalError );
		logger.debug( "final guess   : " + XfmUtils.printArray( invResult ) );
		logger.debug( "final guessXfm: " + XfmUtils.printArray( invResultXfm ) );

		assertTrue( "tolerance met", ( finalError < tolerance  ));
	}

	@Test
	public void testTPSInverse2()
	{
		double[] target = new double[]{ 0.5, 0.5 };
		// double[] guessBase = new double[] { 5.0, 5.0 };
		// double[] guess = new double[ 2 ];

		double[] guess = new double[]{ 5.0, 5.0 };

		double[][] mtx;
		double error = 9999;

		genPtListSimple2d();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts, false );

		double finalError= tps.inverseTol( target, guess, 0.5, 2000 );

		double[] guessXfm = tps.apply( guess );
		logger.debug( "final error   : " + finalError );
		logger.debug( "final guess   : " + XfmUtils.printArray( guess ) );
		logger.debug( "final guessXfm: " + XfmUtils.printArray( guessXfm ) );

		assertEquals( "within tolerance 0.5 x", 0.5, guessXfm[ 0 ], 0.5 );
		assertEquals( "within tolerance 0.5 y", 0.5, guessXfm[ 1 ], 0.5 );

		// tps.inverseTol( target, guess, 0.1, 200 );
		// logger.debug( "final guess: " + XfmUtils.printArray( guess ) );
		//
		// assertEquals( "within tolerance 0.1 x", 0.5, guess[ 0 ], 0.1 );
		// assertEquals( "within tolerance 0.1 y", 0.5, guess[ 1 ], 0.1 );
		//
		// // try for a few different initial guesses
		// for ( int xm = -1; xm <= 1; xm++ )
		// for ( int ym = -1; ym <= 1; ym++ )
		// {
		// System.arraycopy( guessBase, 0, guess, 0, ndims );
		// guess[ 0 ] *= xm;
		// guess[ 1 ] *= ym;
		//
		// tps.inverseTol( target, guess, 0.5, 200 );
		// logger.debug( "final guess: " + XfmUtils.printArray( guess ) );
		//
		// assertEquals( "within tolerance 0.5 x", 0.5, guess[ 0 ], 0.5 );
		// assertEquals( "within tolerance 0.5 y", 0.5, guess[ 1 ], 0.5 );
		//
		// tps.inverseTol( target, guess, 0.1, 2000 );
		// logger.debug( "final guess: " + XfmUtils.printArray( guess ) );
		//
		// assertEquals( "within tolerance 0.1 x", 0.5, guess[ 0 ], 0.1 );
		// assertEquals( "within tolerance 0.1 y", 0.5, guess[ 1 ], 0.1 );
		// }
	}

	@Test
	public void testStepSize()
	{
		double[] target = new double[]{ 0.5, 0.5 };
		double[] guessBase = new double[]{ 2.0, 2.0 };
		double[] guess = new double[ 2 ];
		double[] guessXfm = new double[ 2 ];

		double[][] mtx;
		double error = 9999;
		double tolerance = 0.5;

		genPtListSimple2d();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts, false );

		int xm = 1;
		int ym = 1;

		System.arraycopy( guessBase, 0, guess, 0, ndims );
		guess[ 0 ] *= xm;
		guess[ 1 ] *= ym;

		tps.apply( guess, guessXfm );
		mtx = tps.jacobian( guess );

		TransformInverseGradientDescent inv = new TransformInverseGradientDescent( ndims,
				tps );
		inv.setTarget( target );
		inv.setEstimate( guess );
		inv.setEstimateXfm( guessXfm );
		inv.setJacobian( mtx );
		inv.setEps( 0.001 * tolerance );

		double c = 0.5;
		double beta = 0.75;

		int k = 0;
		mtx = tps.jacobian( guess );
		inv.setJacobian( mtx );

		// inv.oneIteration( false );
		inv.computeDirection();

		// is the Armijo condition satisfied
		boolean isArmijo = inv.armijoCondition( c, 1.0 );

		System.out.println( "isArmijo: " + isArmijo );

	}

	@Test
	public void testTPSInverse()
	{
		double[] target = new double[]{ 0.0, 0.0 };
		double[] guessBase = new double[]{ 5.0, 5.0 };
		double[] guess = new double[ 2 ];

		double[][] mtx;
		double error = 9999;
		int maxIters = 2000;

		genPtListSimple2d();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts, false );

		// try for a few different initial guesses
		int xm = 1;
		int ym = 1;
		System.arraycopy( guessBase, 0, guess, 0, ndims );
		guess[ 0 ] *= xm;
		guess[ 1 ] *= ym;

		double tolerance = 0.5;

		double[] guessXfm = new double[ ndims ];

		tps.apply( guess, guessXfm );
		mtx = tps.jacobian( guess );

		TransformInverseGradientDescent inv = new TransformInverseGradientDescent( ndims,
				tps );
		inv.setTarget( target );
		inv.setEstimate( guess );
		inv.setEstimateXfm( guessXfm );
		inv.setJacobian( mtx );
		inv.setEps( 0.001 * tolerance );

		double c = 0.5;
		double beta = 0.75;

		int k = 0;
		while ( error >= tolerance && k < maxIters )
		{
			mtx = tps.jacobian( guess );
			inv.setJacobian( mtx );
			inv.oneIteration( false );
			inv.computeDirection();

			double t = inv.backtrackingLineSearch( c, beta, 100, 1.0 );
			inv.updateEstimate( t );
			inv.updateError();

			TransformInverseGradientDescent
					.copyVectorIntoArray( inv.getEstimate(), guess );
			tps.apply( guess, guessXfm );

			inv.setEstimateXfm( guessXfm );
			error = inv.getError();
//			System.out.println( "error: " + error );

			k++;
		}

		System.out.println( "error: " + error );
	}

	@Test
	public void testGradientDescentLinear()
	{
		double[] target = new double[]{ 10.0, 10.0 };
		double[] guess = new double[]{ 1.0, 1.0 };
		double[][] mtx = new double[][]{
				{ -1.0, 0.0 },
				{ 0.0, -1.0 } };

		double error = 999;
		// error = testGradientDescent( mtx, target, guess );
		// assertTrue( "is error small", ( error < 1 ) );

		mtx = new double[][]{
				{ -2.0, 0.0 },
				{ 0.0, -2.0 }};
		error = testGradientDescent( mtx, target, guess );
		// assertTrue( "is error small 2", ( error < 1 ) );
	}

	private double testGradientDescent( double[][] mtx, double[] target, double[] guess )
	{
		int ndims = target.length;

		DenseMatrix64F mat = new DenseMatrix64F( mtx );

		DenseMatrix64F guessVec = new DenseMatrix64F( ndims, 1 );
		guessVec.setData( guess );

		DenseMatrix64F xfmVec = new DenseMatrix64F( ndims, 1 );

		CommonOps.mult( mat, guessVec, xfmVec );
		logger.info( "xfmVec:\n" + xfmVec );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims );
		TransformInverseGradientDescent inv = new TransformInverseGradientDescent( ndims,
				tps );
		inv.setJacobian( mtx );
		inv.setTarget( target );
		inv.setEstimate( guess );
		inv.setEstimateXfm( xfmVec.data );
		inv.setStepSize( 1.0 );

		double error = inv.getError();

		int k = 0;
		while ( error > 1 && k < 100 )
		{
			inv.computeDirection();
			inv.updateEstimate( 1.0 );

			CommonOps.mult( mat, guessVec, xfmVec );
			inv.setEstimateXfm( xfmVec.data );

			CommonOps.mult( mat, inv.getEstimate(), xfmVec );
			inv.setEstimateXfm( xfmVec.data );

			error = inv.getError();
			double sqerr = inv.squaredError( xfmVec.data );

			System.out.println( "estimate ( " + k + " ) : "
					+ XfmUtils.printArray( inv.getEstimate().data ) );
			System.out.println( "estimate xfm ( " + k + " ) : "
					+ XfmUtils.printArray( xfmVec.data ) );
			System.out.println( "error ( " + k + " ) : " + sqerr );

			k++;
		}

		return error;
	}

	@Test
	public void testAffineOnly()
	{
		ndims = 3;
		srcPts = new double[][]
		{
				{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 }, // x
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // y
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // z
		};

		final double[][] aff = new double[][]
		{
				{ 0, 1, 0 },
				{ 0, 0, 1 },
				{ 1, 0, 0 }
		};

		N = srcPts[ 0 ].length;
		tgtPts = XfmUtils.genPtListAffine( srcPts, aff );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts, false );

		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < N; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = srcPts[ i ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "pure affine transformation", tgtPts[ i ][ n ], outPt[ i ],
						tol );
			}
		}
	}

	@Test
	public void testIdentitySmall2d()
	{
		final int ndims = 2;

		final double[][] pts = new double[][]
		{
			{ -1, 0, 0 }, // x
			{ -1, 0, 1 }  // y
		};

		final int nL = pts[ 0 ].length;

		final double[][] tpts = XfmUtils.genPtListScale( pts, new double[]{ 1, 1 } );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, pts, tpts );

		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < nL; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = pts[ i ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			System.out.println( outPt.length );
			System.out.println( tpts.length + " x " + tpts[ 0 ].length );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "Identity transformation", tpts[ i ][ n ], outPt[ i ], tol );
			}
		}
	}

	@Test
	public void testIdentitySmall3d()
	{
		final int ndims = 3;

		final double[][] pts = new double[][]
				{
				{ 0, 0, 0, 0, 0, 0, 1, 1 }, // x
				{ 0, 0, 0, 1, 1, 1, 2, 2 }, // y
				{ 0, 1, 2, 0, 1, 2, 0, 1 } // z
		};

		final int nL = pts[ 0 ].length;

		final double[][] tpts = XfmUtils.genPtListScale( pts, new double[]
		{ 2, 3, 0.5 } );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, pts, tpts );

		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < nL; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = pts[ i ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "Identity transformation", tpts[ i ][ n ], outPt[ i ], tol );
			}
		}
	}

	@Test
	public void testIdentity()
	{
		final int ndims = 3;

		final double[][] pts = new double[][]
		{
				{ -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 }, // x
				{ -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // y
				{ -1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // z
		};
		final int nL = pts[ 0 ].length;

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, pts, pts );

		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < nL; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = pts[ i ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "Identity transformation", pts[ i ][ n ], outPt[ i ], tol );
			}
		}

		final ThinPlateR2LogRSplineKernelTransform tpsNA = new ThinPlateR2LogRSplineKernelTransform(
				ndims, pts, pts, false );

		for ( int n = 0; n < nL; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = pts[ i ][ n ];
			}

			final double[] outPt = tpsNA.apply( testPt );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "Identity transformation", pts[ i ][ n ], outPt[ i ], tol );
			}
		}

	}

	@Test
	public void testScale3d()
	{

		final int ndims = 3;
		final double[][] src_simple = new double[][]
		{
				{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 }, // x
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // y
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // z
		};

		final double[][] tgtPtList = XfmUtils.genPtListScale( src_simple, new double[]{ 2, 0.5, 4 } );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, src_simple, tgtPtList );

		double[] srcPt = new double[]{ 0.0f, 0.0f, 0.0f };
		double[] ptXfm = tps.apply( srcPt );
		assertEquals( "scale x1", 0, ptXfm[ 0 ], tol );
		assertEquals( "scale y1", 0, ptXfm[ 1 ], tol );
		assertEquals( "scale z1", 0, ptXfm[ 2 ], tol );

		srcPt = new double[]{ 0.5f, 0.5f, 0.5f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "scale x2", 1.00, ptXfm[ 0 ], tol );
		assertEquals( "scale y2", 0.25, ptXfm[ 1 ], tol );
		assertEquals( "scale z2", 2.00, ptXfm[ 2 ], tol );

		srcPt = new double[]{ 1.0f, 1.0f, 1.0f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "scale x3", 2.0, ptXfm[ 0 ], tol );
		assertEquals( "scale y3", 0.5, ptXfm[ 1 ], tol );
		assertEquals( "scale z3", 4.0, ptXfm[ 2 ], tol );
	}

	@Test
	public void testAffineReasonable()
	{

		genPtListNoAffine1();

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts );
		// tps.setDoAffine(false);

		final double[][] a = tps.getAffine();
		final double[] t = tps.getTranslation();

		System.out.println( "a: " + XfmUtils.printArray( a ) + "\n" );
		System.out.println( "t: " + XfmUtils.printArray( t ) );

		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < 2 * N; n++ )
		{
			for ( int i = 0; i < ndims; i++ )
			{
				testPt[ i ] = srcPts[ i ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			for ( int i = 0; i < ndims; i++ )
			{
				assertEquals( "Identity transformation", tgtPts[ i ][ n ], outPt[ i ],
						tol );
			}
		}
	}

	@Test
	public void testAffineSanity()
	{

		final int ndims = 2;
		final double[][] src = new double[][]
		{
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // x
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // y
		};

		final double[][] tgt = new double[ src.length ][ src[ 0 ].length ];

		for ( int i = 0; i < src.length; i++ )
			for ( int j = 0; j < src[ 0 ].length; j++ )
			{
				tgt[ i ][ j ] = src[ i ][ j ] + Math.random() * 0.1;
			}

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, src, tgt, false );

		// System.out.println(" aMatrix: (Expect all zeros)\n" +
		// printArray(tps.aMatrix) + "\n");
		assertTrue( "aMatrix should be null", tps.aMatrix == null );

		// System.out.println(" bVector: (Expect all zeros)\n" +
		// printArray(tps.bVector) + "\n");
		assertTrue( "bVector should be null", tps.bVector == null );

		// System.out.println(" dMatrix: (Expect non-zero)\n" + tps.dMatrix +
		// "\n");
		boolean isNonZeroMtxElem = false;
		for ( int i = 0; i < tps.dMatrix.getNumElements(); i++ )
		{
			isNonZeroMtxElem = isNonZeroMtxElem || (tps.dMatrix.get( i ) != 0);
		}
		assertTrue( " dMatrix has non-zero element", isNonZeroMtxElem );

	}

	// @Test
	// public void testTransXfmAffineTps(){
	//
	// // int ndims = 2;
	// // float[][] src_simple = new float[][]
	// // {
	// // {-1,-1,-1,1,1,1,2,2,2}, // x
	// // {-1,1,2,-1,1,2,0,1,2}, // y
	// // };
	// //
	// // // target points
	// // float[][] tgt= new float[][]
	// // {
	// // { -0.5f, -0.5f, -0.5f, 1.5f, 1.5f, 1.5f, 2.0f, 2.0f, 2.0f}, // x
	// // { -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f, -0.5f, 1.5f, 2.0f } // y
	// // };
	//
	// int ndims = 2;
	// srcPtsF = new float[][]
	// {
	// {-1,-1,-1,1,1,1,2,2,2}, // x
	// {-1,1,2,-1,1,2,-1,1,2}, // y
	// };
	//
	// // target points
	// tgtPtsF= new float[][]
	// {
	// { 0,0,0, 2,2,2, 3,3,3}, // x
	// { 1,3,4, 1,3,4, 1,3,5 } // y
	// };
	//
	// ThinPlateR2LogRSplineKernelTransformFloatSep tps
	// = new ThinPlateR2LogRSplineKernelTransformFloatSep( ndims, srcPtsF,
	// tgtPtsF);
	//
	// tps.fit();
	//
	// N = srcPtsF[0].length;
	// float[] testPt = new float[ndims];
	// for( int n=0; n<N; n++) {
	//
	// for( int d=0; d<ndims; d++) {
	// testPt[d] = srcPtsF[d][n];
	// }
	//
	// float[] outPt = tps.transform(testPt);
	// logger.debug("point: " + XfmUtils.printArray(testPt) + " -> " +
	// XfmUtils.printArray(outPt));
	// for( int d=0; d<ndims; d++) {
	// assertEquals("translation, use affine", tgtPtsF[d][n], outPt[d], tol);
	// }
	// }
	//
	// }

	@Test
	public void testScale()
	{

		final int ndims = 2;
		srcPts = new double[][]
		{
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // x
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // y
		};

		tgtPts = XfmUtils.genPtListScale( srcPts, new double[]
		{ 2, 0.5 } );

		logger.debug( "srcPts:\n" + XfmUtils.printArray( srcPts ) );
		logger.debug( "\ntgtPts:\n" + XfmUtils.printArray( tgtPts ) );

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, srcPts, tgtPts );

		double[] srcPt = new double[]
		{ 0.0f, 0.0f };
		double[] ptXfm = tps.apply( srcPt );
		assertEquals( "scale x1", 0, ptXfm[ 0 ], tol );
		assertEquals( "scale y1", 0, ptXfm[ 1 ], tol );

		srcPt = new double[]
		{ 0.5f, 0.5f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "scale x2", 1.00, ptXfm[ 0 ], tol );
		assertEquals( "scale y2", 0.25, ptXfm[ 1 ], tol );

		srcPt = new double[]
		{ 1.0f, 1.0f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "scale x3", 2.0, ptXfm[ 0 ], tol );
		assertEquals( "scale y3", 0.5, ptXfm[ 1 ], tol );

		N = srcPts[ 0 ].length;
		final double[] testPt = new double[ ndims ];
		for ( int n = 0; n < N; n++ )
		{

			for ( int d = 0; d < ndims; d++ )
			{
				testPt[ d ] = srcPts[ d ][ n ];
			}

			final double[] outPt = tps.apply( testPt );
			for ( int d = 0; d < ndims; d++ )
			{
				assertEquals( "Identity transformation", tgtPts[ d ][ n ], outPt[ d ],
						tol );
			}
		}
	}

	@Test
	public void testWarp()
	{

		final int ndims = 2;
		final double[][] src_simple = new double[][]
		{
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // x
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // y
		};
		// target points
		final double[][] tgt = new double[][]
		{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0 }, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
		};

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, src_simple, tgt );
		// tps.printLandmarks();

		/* **** PT 2 **** */
		double[] srcPt = new double[]{ 0.0f, 0.0f };
		double[] ptXfm = tps.apply( srcPt );
		assertEquals( "warp x1", -0.5, ptXfm[ 0 ], tol );
		assertEquals( "warp y1", -0.5, ptXfm[ 1 ], tol );

		// check double
		final double[] srcPtF = srcPt.clone();
		final double[] ptXfmF = tps.apply( srcPtF );
		assertEquals( "warp x1 float", -0.5f, ptXfmF[ 0 ], tol );
		assertEquals( "warp y1 float", -0.5f, ptXfmF[ 1 ], tol );

		// double in place
		tps.applyInPlace( srcPt );
		assertEquals( "warp x1 in place", -0.5, srcPt[ 0 ], tol );
		assertEquals( "warp y1 in place", -0.5, srcPt[ 1 ], tol );

		/* **** PT 2 **** */
		srcPt = new double[]{ 0.5f, 0.5f };
		ptXfm = tps.apply( srcPt );

		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals( "warp x2", 0.6241617, ptXfm[ 0 ], tol );
		assertEquals( "warp y2", 0.6241617, ptXfm[ 1 ], tol );

		// double in place 2
		tps.applyInPlace( srcPt );
		assertEquals( "warp x2 in place", 0.6241617, srcPt[ 0 ], tol );
		assertEquals( "warp y2 in place", 0.6241617, srcPt[ 1 ], tol );

		/* **** PT 3 **** */
		srcPt = new double[]{ 1.0f, 1.0f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "warp x3", 1.5, ptXfm[ 0 ], tol );
		assertEquals( "warp y3", 1.5, ptXfm[ 1 ], tol );

		tps.applyInPlace( srcPt );
		assertEquals( "warp x3 in place", 1.5, srcPt[ 0 ], tol );
		assertEquals( "warp y3 in place", 1.5, srcPt[ 1 ], tol );
	}

	@Test
	public void testWeights()
	{

		final int ndims = 2;
		final double[][] src_simple = new double[][]
		{
				{ 0, 0, 0, 1, 1, 1, 2, 2, 2 }, // x
				{ 0, 1, 2, 0, 1, 2, 0, 1, 2 }, // y
		};
		// target points
		final double[][] tgt = new double[][]
		{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0 }, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
		};
		// double[] weights = new double[]
		// { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

		final double[] weights = new double[]
				{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0 };

		final ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform(
				ndims, src_simple, tgt, weights );
		// tps.printLandmarks();

		double[] srcPt = new double[]{ 0.0f, 0.0f };
		double[] ptXfm = tps.apply( srcPt );
		assertEquals( "warp x1", -0.5, ptXfm[ 0 ], tol );
		assertEquals( "warp y1", -0.5, ptXfm[ 1 ], tol );

		srcPt = new double[]{ 0.5f, 0.5f };
		ptXfm = tps.apply( srcPt );

		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals( "warp x2", 0.6241617, ptXfm[ 0 ], tol );
		assertEquals( "warp y2", 0.6241617, ptXfm[ 1 ], tol );

		srcPt = new double[]{ 1.0f, 1.0f };
		ptXfm = tps.apply( srcPt );
		assertEquals( "warp x3", 1.5, ptXfm[ 0 ], tol );
		assertEquals( "warp y3", 1.5, ptXfm[ 1 ], tol );

	}

}

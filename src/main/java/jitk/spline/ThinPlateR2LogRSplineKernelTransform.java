package jitk.spline;

import java.util.Arrays;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;

import mpicbg.models.CoordinateTransform;

/**
 * Implements a thin plate spline transform.
 * Began as a port of itk's itkKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms, “A physics-based
 * coordinate transformation for 3-D image matching.,” IEEE Trans. Med. Imaging,
 * vol. 16, no. 3, pp. 317–28, Jun. 1997.
 *
 * @author Kitware (ITK)
 * @author John Bogovic
 *
 */
public class ThinPlateR2LogRSplineKernelTransform implements CoordinateTransform
{

	private static final long serialVersionUID = -972934724062617822L;

	protected final int ndims;

	protected DenseMatrix64F dMatrix;

	protected double[][] aMatrix;

	protected double[] bVector;

	protected double stiffness = 0.0; // reasonable values take the range [0.0, 0.5]

	protected boolean wMatrixComputeD = false;

	protected boolean computeAffine = true;

	protected int nLandmarks;

	protected final double[][] sourceLandmarks;

	protected double[] weights; // TODO: make the weights do something :-P

	protected static final double EPS = 1e-8;

	protected static Logger logger = LogManager.getLogger( 
			ThinPlateR2LogRSplineKernelTransformFinal.class.getName() );

	/*
	 * Constructs an identity thin plate spline transform
	 */
	public ThinPlateR2LogRSplineKernelTransform( final int ndims )
	{
		this.ndims = ndims;
		sourceLandmarks = null;
		nLandmarks = 0;
	}

	public ThinPlateR2LogRSplineKernelTransform( final int ndims,
			final double[][] srcPts, final double[][] tgtPts )
	{
		this( ndims, srcPts, tgtPts, true );
	}

	public ThinPlateR2LogRSplineKernelTransform( final int ndims,
			final float[][] srcPts, final float[][] tgtPts )
	{
		this( ndims, srcPts, tgtPts, true );
	}

	/*
	 * Constructor with point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform( final int ndims,
			final double[][] srcPts, final double[][] tgtPts, boolean computeAffine )
	{
		this.ndims = ndims;
		this.sourceLandmarks = srcPts;
		this.computeAffine = computeAffine;

		if ( sourceLandmarks != null && sourceLandmarks.length > 0 )
			nLandmarks = srcPts[ 0 ].length;
		else
			nLandmarks = 0;

		computeW( buildDisplacements( tgtPts ) );
	}

	/*
	 * Constructor with point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform( final int ndims, final float[][] srcPts,
			final float[][] tgtPts, boolean computeAffine )
	{
		this.ndims = ndims;
		this.computeAffine = computeAffine;
		sourceLandmarks = new double[ ndims ][ nLandmarks ];

		if ( sourceLandmarks != null && sourceLandmarks.length > 0 )
			nLandmarks = srcPts[ 0 ].length;
		else
			nLandmarks = 0;

		for ( int i = 0; i < nLandmarks; ++i )
		{
			for ( int d = 0; d < ndims; ++d )
			{
				sourceLandmarks[ d ][ i ] = srcPts[ d ][ i ];
			}
		}

		computeW( buildDisplacements( tgtPts ) );
	}

	/*
	 * Constructor with weighted point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform( final int ndims,
			final double[][] srcPts, final double[][] tgtPts, final double[] weights )
	{
		this( ndims, srcPts, tgtPts );
		setWeights( weights );
	}

	/**
	 * Constructor with transformation parameters. aMatrix and bVector are
	 * allowed to be null
	 */
	public ThinPlateR2LogRSplineKernelTransform( final double[][] srcPts,
			final double[][] aMatrix, final double[] bVector,
			final double[] dMatrixData )
	{
		ndims = srcPts.length;

		if ( srcPts != null && srcPts.length > 0 )
			nLandmarks = srcPts[ 0 ].length;
		else
			nLandmarks = 0;

		this.sourceLandmarks = srcPts;
		this.aMatrix = aMatrix;
		this.bVector = bVector;

		dMatrix = new DenseMatrix64F( ndims, nLandmarks );
		dMatrix.setData( dMatrixData );
	}

	public int getNumLandmarks()
	{
		return this.nLandmarks;
	}

	public int getNumDims()
	{
		return ndims;
	}

	public double[][] getSourceLandmarks()
	{
		return sourceLandmarks;
	}

	public double[][] getAffine()
	{
		return aMatrix;
	}

	public double[] getTranslation()
	{
		return bVector;
	}

	public double[] getKnotWeights()
	{
		return dMatrix.getData();
	}

	/**
	 * Sets the weights. Checks that the length matches the number of landmarks.
	 */
	private void setWeights( final double[] weights )
	{
		// make sure the length matches number
		// of landmarks
		if ( weights == null ) { return; }
		if ( weights.length != this.nLandmarks )
		{
			this.weights = weights;
		}
		else
		{
			logger.error( "weights have length (" + weights.length
					+ ") but tmust have length equal to number of landmarks "
					+ this.nLandmarks );
		}
	}

	public void setDoAffine( final boolean estimateAffine )
	{
		this.computeAffine = estimateAffine;
	}

	private void initMatrices(DenseMatrix64F kMatrix, DenseMatrix64F lMatrix, DenseMatrix64F pMatrix, DenseMatrix64F wMatrix, DenseMatrix64F yMatrix)
	{
		dMatrix = new DenseMatrix64F( ndims, nLandmarks );
		kMatrix = new DenseMatrix64F( ndims * nLandmarks, ndims * nLandmarks );

		if ( computeAffine )
		{
			aMatrix = new double[ ndims ][ ndims ];
			bVector = new double[ ndims ];

			pMatrix = new DenseMatrix64F( ( ndims * nLandmarks ),
					( ndims * ( ndims + 1 ) ) );
			lMatrix = new DenseMatrix64F( ndims * ( nLandmarks + ndims + 1 ),
					ndims * ( nLandmarks + ndims + 1 ) );
			wMatrix = new DenseMatrix64F( ( ndims * nLandmarks ) + ndims * ( ndims + 1 ), 1 );
			yMatrix = new DenseMatrix64F( ndims * ( nLandmarks + ndims + 1 ), 1 );
		}
		else
		{
			// we dont need the P matrix and L can point
			// directly to K rather than itself being initialized

			// the W matrix won't hold the affine component
			wMatrix = new DenseMatrix64F( ndims * nLandmarks, 1 );
			yMatrix = new DenseMatrix64F( ndims * nLandmarks, 1 );
		}
	}

	protected DenseMatrix64F computeReflexiveG()
	{
		DenseMatrix64F gMatrix = new DenseMatrix64F( ndims, ndims );
		CommonOps.fill( gMatrix, 0 );
		for ( int i = 0; i < ndims; i++ )
		{
			gMatrix.set( i, i, stiffness );
		}
		return gMatrix;
	}

	protected double normSqrd( final double[] v )
	{
		double nrm = 0;
		for ( int i = 0; i < v.length; i++ )
		{
			nrm += v[ i ] * v[ i ];
		}
		return nrm;
	}

	public DenseMatrix64F buildDisplacements( double[][] targetLandmarks )
	{
		DenseMatrix64F yMatrix;
		if ( computeAffine )
			yMatrix = new DenseMatrix64F( ndims * (nLandmarks + ndims + 1), 1 );
		else
			yMatrix = new DenseMatrix64F( ndims * nLandmarks, 1 );


		// for (int i = 0; i < nLandmarks; i++) {
		int i = 0;
		while ( i < nLandmarks )
		{
			for ( int j = 0; j < ndims; j++ )
			{
				yMatrix.set( i * ndims + j, 0,
						(targetLandmarks[ j ][ i ] - sourceLandmarks[ j ][ i ]) );
			}
			i++;
		}
		if ( computeAffine )
		{
			for ( i = 0; i < ndims * (ndims + 1); i++ )
			{
				yMatrix.set( nLandmarks * ndims + i, 0, 0 );
			}
		}

		return yMatrix;
	}
	
	public DenseMatrix64F buildDisplacements( float[][] targetLandmarks )
	{
		DenseMatrix64F yMatrix;
		if ( computeAffine )
			yMatrix = new DenseMatrix64F( ndims * (nLandmarks + ndims + 1), 1 );
		else
			yMatrix = new DenseMatrix64F( ndims * nLandmarks, 1 );


		// for (int i = 0; i < nLandmarks; i++) {
		int i = 0;
		while ( i < nLandmarks )
		{
			for ( int j = 0; j < ndims; j++ )
			{
				yMatrix.set( i * ndims + j, 0,
						(targetLandmarks[ j ][ i ] - sourceLandmarks[ j ][ i ]) );
			}
			i++;
		}
		if ( computeAffine )
		{
			for ( i = 0; i < ndims * (ndims + 1); i++ )
			{
				yMatrix.set( nLandmarks * ndims + i, 0, 0 );
			}
		}

		return yMatrix;
	}

	/**
	 * The main workhorse method.
	 * <p>
	 * Implements Equation (5) in Davis et al. and calls reorganizeW.
	 *
	 */
	protected void computeW( final DenseMatrix64F yMatrix )
	{

		final DenseMatrix64F kMatrix;
		final DenseMatrix64F pMatrix;
		final DenseMatrix64F wMatrix;

		/**
		 * INITIALIZE MATRICES
		 */
		dMatrix = new DenseMatrix64F( ndims, nLandmarks );
		kMatrix = new DenseMatrix64F( ndims * nLandmarks, ndims * nLandmarks );

		if ( computeAffine )
		{
			aMatrix = new double[ ndims ][ ndims ];
			bVector = new double[ ndims ];

			pMatrix = new DenseMatrix64F( ( ndims * nLandmarks ),
					( ndims * ( ndims + 1 ) ) );

			wMatrix = new DenseMatrix64F( ( ndims * nLandmarks ) + ndims * ( ndims + 1 ), 1 );
			
		}
		else
		{
			// we dont need the P matrix and L can point
			// directly to K rather than itself being initialized

			// the W matrix won't hold the affine component
			wMatrix = new DenseMatrix64F( ndims * nLandmarks, 1 );
			pMatrix = null;
		}
		
//		gMatrix = new DenseMatrix64F( ndims, ndims );
		
		final DenseMatrix64F lMatrix = computeL( kMatrix, pMatrix );

		final LinearSolver< DenseMatrix64F > solver;
		if ( nLandmarks < ndims * ndims )
		{
			logger.debug( "pseudo inverse solver" );
			solver = LinearSolverFactory.pseudoInverse( false );
		}
		else
		{
			logger.debug( "linear solver" );
			solver = LinearSolverFactory.linear( lMatrix.numCols );
		}

		// solve linear system
		solver.setA( lMatrix );
		solver.solve( yMatrix, wMatrix );

		reorganizeW( wMatrix );

	}

	protected DenseMatrix64F computeL(DenseMatrix64F kMatrix, DenseMatrix64F pMatrix)
	{
		computeK( kMatrix );

		// fill P matrix if the affine parameters need to be computed
		if ( computeAffine )
		{
			computeP( pMatrix );

			DenseMatrix64F lMatrix = new DenseMatrix64F( ndims * ( nLandmarks + ndims + 1 ),
					ndims * ( nLandmarks + ndims + 1 ) );
			
			CommonOps.insert( kMatrix, lMatrix, 0, 0 );
			CommonOps.insert( pMatrix, lMatrix, 0, kMatrix.getNumCols() );
			CommonOps.transpose( pMatrix );

			CommonOps.insert( pMatrix, lMatrix, kMatrix.getNumRows(), 0 );
			CommonOps.insert( kMatrix, lMatrix, 0, 0 );
			// P matrix should be zero if points are already affinely aligned
			// bottom left O2 is already zeros after initializing 'lMatrix'
			
			return lMatrix;
		}
		else
		{
			// in this case the L matrix
			// consists only of the K block.
			return kMatrix;
		}

	}

	protected void computeP(DenseMatrix64F pMatrix)
	{
		final DenseMatrix64F tmp = new DenseMatrix64F( ndims, ndims );
		final DenseMatrix64F I = new DenseMatrix64F( ndims, ndims );
		CommonOps.setIdentity( I );
		
		int i = 0;
		while ( i < nLandmarks )
		{
			for ( int d = 0; d < ndims; d++ )
			{
				CommonOps.scale( sourceLandmarks[ d ][ i ], I, tmp );
				CommonOps.insert( tmp, pMatrix, i * ndims, d * ndims );
			}
			CommonOps.insert( I, pMatrix, i * ndims, ndims * ndims );
			i++;
		}
	}

	/**
	 * Builds the K matrix from landmark points and G matrix.
	 * @param kMatrix 
	 */
	protected void computeK(DenseMatrix64F kMatrix)
	{
		final double[] res = new double[ ndims ];

		int i = 0;
		
		final DenseMatrix64F Gbase = computeReflexiveG();
		final DenseMatrix64F G = Gbase.copy();
		
		while ( i < nLandmarks )
		{
			CommonOps.insert( Gbase, kMatrix, i * ndims, i * ndims );

			int j = i + 1;
			while ( j < nLandmarks )
			{
				srcPtDisplacement( i, j, res );
				computeG( res, G );

				CommonOps.insert( G, kMatrix, i * ndims, j * ndims );
				CommonOps.insert( G, kMatrix, j * ndims, i * ndims );

				j++;
			}
			i++;
		}
	}

	/**
	 * Copies data from the W matrix to the D, A, and b matrices which represent
	 * the deformable, affine and translational portions of the transformation,
	 * respectively.
	 * @param wMatrix  
	 */
	protected void reorganizeW (DenseMatrix64F wMatrix )
	{
		// the deformable (non-affine) part of the transform
		int ci = 0;
		int i = 0;
		while ( i < nLandmarks )
		{
			for ( int d = 0; d < ndims; d++ )
			{
				dMatrix.set( d, i, wMatrix .get( ci, 0 ) );
				ci++;
			}
			i++;
		}

		// the affine part of the transform
		if ( computeAffine )
		{
			// the affine part of the transform
			for ( int j = 0; j < ndims; j++ )
				for ( i = 0; i < ndims; i++ )
				{
					aMatrix[ i ][ j ] = wMatrix.get( ci, 0 );
					ci++;
				}

			// the translation part of the transform
			for ( int k = 0; k < ndims; k++ )
			{
				bVector[ k ] = wMatrix.get( ci, 0 );
				ci++;
			}
		}

		wMatrix = null;
	}

	public void computeG( final double[] pt, final DenseMatrix64F mtx )
	{

		final double r = Math.sqrt( normSqrd( pt ) );
		final double nrm = r2Logr( r );

		CommonOps.setIdentity( mtx );
		CommonOps.scale( nrm, mtx );
	}

	public void computeG( final double[] pt, final DenseMatrix64F mtx,
			final double w )
	{

		computeG( pt, mtx );
		CommonOps.scale( w, mtx );
	}

	public void computeDeformationContribution( final double[] thispt,
			final double[] result )
	{

		final double[] tmpDisplacement = new double[ ndims ];
		for ( int i = 0; i < ndims; ++i )
		{
			result[ i ] = 0;
			tmpDisplacement[ i ] = 0;
		}

		int di = 0;
		for ( int lnd = 0; lnd < nLandmarks; lnd++ )
		{
			srcPtDisplacement( lnd, thispt, tmpDisplacement );
			final double nrm = r2Logr( Math.sqrt( normSqrd( tmpDisplacement ) ) );

			for ( int d = 0; d < ndims; d++ )
				result[ d ] += nrm * dMatrix.get( d, di );

			di++;
		}
	}

	/**
	 * The derivative of component j of the output point, with respect to x_d
	 * (the dth component of the vector) is:
	 *
	 * \sum_i D[ d, l ] G'( r ) ( x_j - l_ij ) / ( sqrt( N_l(p) ))
	 *
	 * where: D is the D matrix i indexes the landmarks N_l(p) is the squared
	 * euclidean norm between landmark l and point p l_ij is the jth component
	 * of landmark i G' is the derivative of the kernel function. for a TPS, the
	 * kernel function is (r^2)logr, the derivative wrt r of which is: r( 2logr
	 * + 1 )
	 *
	 * See the documentation for a derivation.
	 *
	 * @param p
	 *            The point at which to evaluate the derivative
	 * @return
	 */
	public double[][] r2LogrDerivative( final double[] p )
	{
		// derivativeMatrix[j][d] gives the derivative of component j with
		// respect to component d
		final double[][] derivativeMatrix = new double[ ndims ][ ndims ];

		final double[] tmpDisplacement = new double[ ndims ];
		Arrays.fill( tmpDisplacement, 0 );

		int lmi = 0; // landmark index for active points
		for ( int lnd = 0; lnd < nLandmarks; lnd++ )
		{
			srcPtDisplacement( lnd, p, tmpDisplacement );

			final double r2 = normSqrd( tmpDisplacement ); // squared radius
			final double r = Math.sqrt( r2 ); // radius

			// TODO if r2 is small or zero, there will be problems - put a
			// check in.
			// The check is below.
			// The continue statement is akin to term1 = 0.0.
			// This should be correct, but needs a double-checking
			final double term1;
			if ( r < EPS )
				continue;
			else
				term1 = r * (2 * Math.log( r ) + 1) / Math.sqrt( r2 );

			for ( int d = 0; d < ndims; d++ )
			{
				for ( int j = 0; j < ndims; j++ )
				{
					final double multiplier = term1 * (-tmpDisplacement[ j ]);
					derivativeMatrix[ j ][ d ] += multiplier * dMatrix.get( d, lmi );
				}
			}
			lmi++;
		}

		return derivativeMatrix;
	}

	/**
	 * Computes the jacobian of this tranformation around the point p.
	 * <p>
	 * The result is stored in a new double array where element [ i ][ j ] gives
	 * the derivative of variable i with respect to variable j
	 *
	 * @param p
	 *            the point
	 * @return the jacobian array
	 */
	public double[][] jacobian( final double[] p )
	{
		final double[][] D = r2LogrDerivative( p );

		if ( aMatrix != null )
		{
			for ( int i = 0; i < ndims; i++ )
				for ( int j = 0; j < ndims; j++ )
					if ( i == j )
						D[ i ][ j ] += 1 + aMatrix[ i ][ j ];
					else
						D[ i ][ j ] += aMatrix[ i ][ j ];
		}

		return D;
	}

	public void stepInDerivativeDirection( final double[][] derivative, final double[] start, final double[] dest, final double stepLength )
	{
		for ( int i = 0; i < ndims; i++ )
		{
			dest[ i ] = start[ i ];
			for ( int j = 0; j < ndims; j++ )
			{
				dest[ i ] = derivative[ i ][ j ] * stepLength;
			}
		}
	}

	public void printXfmBacks2d( final int maxx, final int maxy, final int delx, final int dely )
	{
		final double[] pt = new double[ 2 ];
		final double[] result = new double[ 2 ];
		for ( int x = 0; x < maxx; x += delx )
			for ( int y = 0; y < maxy; y += dely )
			{
				pt[ 0 ] = x;
				pt[ 1 ] = y;

				this.apply( pt, result );
				System.out.println( "( " + x + ", " + y + " )  ->  ( " + result[ 0 ] + ", " + result[ 0 ] + " )" );
			}
	}

	/**
	 * Transforms the input point according to the affine part of the thin plate
	 * spline stored by this object.
	 *
	 * @param pt
	 *            the point to be transformed
	 * @return the transformed point
	 */
	public double[] transformPointAffine( final double[] pt )
	{

		final double[] result = new double[ ndims ];
		// affine part
		for ( int i = 0; i < ndims; i++ )
		{
			for ( int j = 0; j < ndims; j++ )
			{
				result[ i ] += aMatrix[ i ][ j ] * pt[ j ];
			}
		}

		// translational part
		for ( int i = 0; i < ndims; i++ )
		{
			result[ i ] += bVector[ i ] + pt[ i ];
		}

		return result;
	}

	public void apply( final double[] pt, final double[] result )
	{
		apply( pt, result, false );
	}

	/**
	 * Transform a source vector pt into a target vector result. pt and result
	 * must NOT be the same vector.
	 * 
	 * @param pt
	 * @param result
	 */
	public void apply( final double[] pt, final double[] result, final boolean debug )
	{
		if( dMatrix == null )
		{
			for ( int j = 0; j < ndims; j++ )
				result[ j ] = pt[ j ];

			return;
		}

		computeDeformationContribution( pt, result );

		if ( aMatrix != null )
		{
			// affine part
			for ( int i = 0; i < ndims; i++ )
				for ( int j = 0; j < ndims; j++ )
				{
					result[ i ] += aMatrix[ i ][ j ] * pt[ j ];
				}
		} else
		{
			for ( int i = 0; i < ndims; i++ )
			{
				result[ i ] += pt[ i ];
			}
		}

		if ( bVector != null )
		{
			// translational part
			for ( int i = 0; i < ndims; i++ )
			{
				result[ i ] += bVector[ i ] + pt[ i ];
			}
		}

	}

	/**
	 * Transforms the input point according to the thin plate spline stored by
	 * this object.
	 *
	 * @param pt
	 *            the point to be transformed
	 * @return the transformed point
	 */
	@Override
	public double[] apply( final double[] pt )
	{
		final double[] result = new double[ ndims ];

		apply( pt, result );

		return result;
	}

	/**
	 * Transform pt in place.
	 *
	 * @param pt
	 */
	@Override
	public void applyInPlace( final double[] pt )
	{

		final double[] tmp = new double[ ndims ];
		apply( pt, tmp );

		for ( int i = 0; i < ndims; ++i )
		{
			pt[ i ] = tmp[ i ];
		}
	}

	/**
	 * Determine if a point whose inverse we want is close to a landmark. If so,
	 * return the index of that landmark, and use that to help.
	 *
	 * @param pt
	 * @param tolerance
	 * @return a pair containing the closest landmark point and its squared
	 *         distance to that landmark
	 */
	public IndexDistancePair closestTargetLandmarkAndDistance( final double[] target )
	{
		int idx = -1;
		double distSqr = Double.MAX_VALUE;
		double thisDist = 0.0;

		final double[] err = new double[ this.ndims ];

		for ( int l = 0; l < this.nLandmarks; l++ )
		{
			tgtPtDisplacement( l, target, err );
			thisDist = normSqrd( err );

			if ( thisDist < distSqr )
			{
				distSqr = thisDist;
				idx = l;
			}
		}

		return new IndexDistancePair( idx, distSqr );
	}

	private static class IndexDistancePair
	{
		final int index;
		final double distance;

		public IndexDistancePair( final int i, final double d )
		{
			this.index = i;
			this.distance = d;
		}

	}

	public double[] initialGuessAtInverse( final double[] target, final double tolerance )
	{
		final IndexDistancePair lmAndDist = closestTargetLandmarkAndDistance( target );
		logger.trace( "nearest landmark error: " + lmAndDist.distance );

		double[] initialGuess;
		final int idx = lmAndDist.index;
		logger.trace( "initial guess by landmark: " + idx );

		initialGuess = new double[ ndims ];
		for ( int i = 0; i < ndims; i++ )
			initialGuess[ i ] = sourceLandmarks[ i ][ idx ];

		logger.trace( "initial guess by affine " );
		final double[] initialGuessAffine = inverseGuessAffineInv( target );

		final double[] resL = apply( initialGuess );
		final double[] resA = apply( initialGuessAffine );

		for ( int i = 0; i < ndims; i++ )
		{
			resL[ i ] -= target[ i ];
			resA[ i ] -= target[ i ];
		}

		final double errL = normSqrd( resL );
		final double errA = normSqrd( resA );

		logger.trace( "landmark guess error: " + errL );
		logger.trace( "affine guess error  : " + errA );

		if ( errA < errL )
		{
			logger.trace( "Using affine initialization" );
			initialGuess = initialGuessAffine;
		}
		else
		{
			logger.trace( "Using landmark initialization" );
		}

		return initialGuess;
	}

	public double[] inverseGuessAffineInv( final double[] target )
	{
		// Here, mtx is A + I
		final DenseMatrix64F mtx = new DenseMatrix64F( ndims + 1, ndims + 1 );
		final DenseMatrix64F vec = new DenseMatrix64F( ndims + 1, 1 );
		for ( int i = 0; i < ndims; i++ )
		{
			for ( int j = 0; j < ndims; j++ )
			{
				if ( i == j )
					mtx.set( i, j, 1 + aMatrix[ i ][ j ] );
				else
					mtx.set( i, j, aMatrix[ i ][ j ] );
			}
			mtx.set( i, ndims, bVector[ i ] );
			vec.set( i, 0, target[ i ] );
		}
		mtx.set( ndims, ndims, 1.0 );
		vec.set( ndims, 0, 1.0 );

		final DenseMatrix64F res = new DenseMatrix64F( ndims + 1, 1 );

		CommonOps.solve( mtx, vec, res );

		final DenseMatrix64F test = new DenseMatrix64F( ndims + 1, 1 );
		CommonOps.mult( mtx, res, test );

		logger.trace( "test result: " + test );

		final double[] resOut = new double[ ndims ];
		System.arraycopy( res.data, 0, resOut, 0, ndims );

		return resOut;
	}

	public int inverseTol( final double[] pt, final double[] guess, final double tolerance, final int maxIters )
	{
		// TODO - have a flag in the apply method to also return the derivative
		// if requested
		// to prevent duplicated effort

		final double c = 0.0001;
		final double beta = 0.7;
		double error = 999 * tolerance;
		double[][] mtx;
		final double[] guessXfm = new double[ ndims ];

		apply( guess, guessXfm );
		mtx = jacobian( guess );

		final TransformInverseGradientDescent inv = new TransformInverseGradientDescent( ndims, this );
		inv.setTarget( pt );
		inv.setEstimate( guess );
		inv.setEstimateXfm( guessXfm );
		inv.setJacobian( mtx );

		error = inv.getError();
		double t0 = error;
		double t = 1.0;

		int k = 0;
		while ( error >= tolerance && k < maxIters )
		{
			logger.trace( "iteration : " + k );

			mtx = jacobian( guess );
			inv.setJacobian( mtx );
			inv.computeDirection();

			logger.trace( "initial step size: " + t0 );
			t = inv.backtrackingLineSearch( c, beta, 15, t0 );
			logger.trace( "final step size  : " + t );

			if ( t == 0.0 )
				break;

			inv.updateEstimate( t );
			inv.updateError();

			TransformInverseGradientDescent.copyVectorIntoArray( inv.getEstimate(), guess );
			apply( guess, guessXfm );

			t0 = error;

			inv.setEstimateXfm( guessXfm );
			error = inv.getError();

			logger.trace( "guess       : " + XfmUtils.printArray( guess ) );
			logger.trace( "guessXfm    : " + XfmUtils.printArray( guessXfm ) );
			logger.trace( "error vector: " + XfmUtils.printArray( inv.getErrorVector().data ) );
			logger.trace( "error       : " + NormOps.normP2( inv.getErrorVector() ) );
			logger.trace( "abs error   : " + error );
			logger.trace( "" );

			k++;
		}
		return k;
	}

	/**
	 * Computes the displacement between the i^th and j^th source points.
	 *
	 * Stores the result in the input array 'res' Does not validate inputs.
	 */
	protected void srcPtDisplacement( final int i, final int j,
			final double[] res )
	{
		for ( int d = 0; d < ndims; d++ )
		{
			res[ d ] = sourceLandmarks[ d ][ i ] - sourceLandmarks[ d ][ j ];
		}
	}

	/**
	 * Computes the displacement between the i^th source point and the input
	 * point.
	 *
	 * Stores the result in the input array 'res'. Does not validate inputs.
	 */
	protected void srcPtDisplacement( final int i, final double[] pt,
			final double[] res )
	{
		for ( int d = 0; d < ndims; d++ )
		{
			res[ d ] = sourceLandmarks[ d ][ i ] - pt[ d ];
		}
	}

	/**
	 * Computes the displacement between the i^th source point and the input
	 * point.
	 *
	 * Stores the result in the input array 'res'. Does not validate inputs.
	 */
	protected void tgtPtDisplacement( final int i, final double[] pt,
			final double[] res )
	{
		apply( pt, res );
		for ( int d = 0; d < ndims; d++ )
		{
			res[ d ]-= pt[ d ];
		}
	}

	private static double r2Logr( final double r )
	{
		double nrm = 0;
		if ( r > EPS )
		{
			nrm = r * r * Math.log( r );
		}
		return nrm;
	}

}

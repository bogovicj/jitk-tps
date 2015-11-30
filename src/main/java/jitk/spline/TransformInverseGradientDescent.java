package jitk.spline;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class TransformInverseGradientDescent
{
	int ndims;

	DenseMatrix64F mtx;

	DenseMatrix64F dir; // descent direction

	DenseMatrix64F errorV; // error vector

	DenseMatrix64F estimate; // current estimate

	DenseMatrix64F estimateXfm; // current estimateXfm

	DenseMatrix64F target;

	double error = 9999.0;

	double stepSz = 1.0;

	int maxIters = 20;

	double eps = 1e-6;

	protected static Logger logger = LogManager.getLogger(
			TransformInverseGradientDescent.class.getName() );

	public TransformInverseGradientDescent( int ndims )
	{
		this.ndims = ndims;
		dir = new DenseMatrix64F( ndims, 1 );
		errorV = new DenseMatrix64F( ndims, 1 );
	}

	public void setEps( double eps )
	{
		this.eps = eps;
	}

	public void setStepSize( double stepSize )
	{
		stepSz = stepSize;
	}

	public void setGradientMatrix( double[][] mtx )
	{
		this.mtx = new DenseMatrix64F( mtx );
		logger.debug( "setGradientMatrix:\n" + this.mtx );
	}

	public void setTarget( double[] tgt )
	{
		this.target = new DenseMatrix64F( ndims, 1 );
		target.setData( tgt );
	}

	public DenseMatrix64F getErrorVector()
	{
		return errorV;
	}

	public void setEstimate( double[] est )
	{
		this.estimate = new DenseMatrix64F( ndims, 1 );
		estimate.setData( est );
	}

	public void setEstimateXfm( double[] est )
	{
		this.estimateXfm = new DenseMatrix64F( ndims, 1 );
		estimateXfm.setData( est );
		updateError();
	}

	public DenseMatrix64F getEstimate()
	{
		logger.debug( "getEstimate:\n" + estimate );
		return estimate;
	}

	public double getError()
	{
		return error;
	}

	public void oneIteration()
	{
		oneIteration( true );
	}

	public void oneIteration( boolean updateError )
	{
		// at this point, we need a target, an estimate, and a derivative matrix
		computeDirection();
		updateEstimate( stepSz );
		if ( updateError )
			updateError();
	}

	/**
	 * Computes 2A^T(Ax - b ) using the current matrix as A, the current error
	 * vector as b, and the current estimate as x
	 */
	private void computeDirection()
	{
		DenseMatrix64F tmp = new DenseMatrix64F( ndims, 1 );

		CommonOps.mult( mtx, estimate, tmp );
		CommonOps.subEquals( tmp, errorV );
		// now tmp contains Ax-b

		// performs dir = 2M^T( tmp ) = 2M^T( Mx - b )
		CommonOps.multTransA( 2, mtx, tmp, dir );

		// normalize
//		double norm = NormOps.normP2( dir );
//		logger.debug( "" );
//		if ( norm > eps )
//		{
//			CommonOps.scale( 1 / norm, dir );
//			logger.debug( "norm big enough: " + norm );
//		}
//		else
//		{
//			logger.debug( "norm small: " + norm );
//			CommonOps.fill( dir, 0.0 );
//		}

		logger.debug( "new direction\n" + dir );

	}

	private void updateEstimate( double stepSize )
	{
		logger.debug( "step size: " + stepSize );
		logger.debug( "estimate:\n" + estimate );

		double norm = NormOps.normP2( dir );
		logger.debug( "norm: " + norm );

		// go in the negative gradient direction to minimize cost
		//
		if ( norm > stepSize )
		{
			CommonOps.scale( -stepSize / norm, dir );
			logger.debug( "norm big enough: " + norm );
		}

		CommonOps.addEquals( estimate, dir );

		logger.debug( "new estimate:\n" + estimate );
	}

	private void updateError()
	{
		if ( estimate == null || target == null )
		{
			System.err.println( "WARNING: Call to updateError with null target or estimate" );
			return;
		}

		CommonOps.sub( estimate, target, errorV );

		logger.debug( "updateError, estimate:\n" + estimate );
		logger.debug( "updateError, target  :\n" + target );
		logger.debug( "updateError, error   :\n" + errorV );

		// set scalar error equal to max of component-wise errors
		error = Math.abs( errorV.get( 0 ) );
		for ( int i = 1; i < ndims; i++ )
		{
			if ( Math.abs( errorV.get( i ) ) > error )
				error = Math.abs( errorV.get( i ) );
		}

	}

	public static void copyVectorIntoArray( DenseMatrix64F vec, double[] array )
	{
		System.arraycopy( vec.data, 0, array, 0, vec.getNumElements() );
	}

}

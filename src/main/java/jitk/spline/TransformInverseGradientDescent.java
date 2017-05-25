package jitk.spline;

import mpicbg.models.CoordinateTransform;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class TransformInverseGradientDescent
{
	int ndims;

	CoordinateTransform xfm;

	DenseMatrix64F jacobian;

	DenseMatrix64F directionalDeriv; // derivative in direction of dir (the
										// descent direction )

	DenseMatrix64F descentDirectionMag; // computes dir^T directionalDeriv
										// (where dir^T is often
										// -directionalDeriv)

	DenseMatrix64F dir; // descent direction

	DenseMatrix64F errorV; // error vector ( errorV = target - estimateXfm )

	DenseMatrix64F estimate; // current estimate

	DenseMatrix64F estimateXfm; // current estimateXfm

	DenseMatrix64F target;

	double error = 9999.0;

	double stepSz = 1.0;

	int maxIters = 20;

	double eps = 1e-6;

	double beta = 0.7;

	protected static Logger logger = LogManager.getLogger(
			TransformInverseGradientDescent.class.getName() );

	public TransformInverseGradientDescent( int ndims, CoordinateTransform xfm )
	{
		this.ndims = ndims;
		this.xfm = xfm;
		dir = new DenseMatrix64F( ndims, 1 );
		errorV = new DenseMatrix64F( ndims, 1 );
		directionalDeriv = new DenseMatrix64F( ndims, 1 );
		descentDirectionMag = new DenseMatrix64F( 1, 1 );
	}

	public void setEps( double eps )
	{
		this.eps = eps;
	}

	public void setStepSize( double stepSize )
	{
		stepSz = stepSize;
	}

	public void setJacobian( double[][] mtx )
	{
		this.jacobian = new DenseMatrix64F( mtx );
		logger.trace( "setJacobian:\n" + this.jacobian );
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

	public DenseMatrix64F getDirection()
	{
		return dir;
	}

	public DenseMatrix64F getJacobian()
	{
		return jacobian;
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
	public void computeDirectionSteepest()
	{
		DenseMatrix64F tmp = new DenseMatrix64F( ndims, 1 );

		logger.trace( "\nerrorV:\n" + errorV );

		CommonOps.mult( jacobian, estimate, tmp );
		// TODO this line is wrong isnt it
		CommonOps.subEquals( tmp, errorV );

		// now tmp contains Ax-b
		CommonOps.multTransA( 2, jacobian, tmp, dir );

		// normalize dir
		double norm = NormOps.normP2( dir );
		// normalize
		// TODO put in a check if norm is too small
		CommonOps.divide( norm, dir );

		// compute the directional derivative
		CommonOps.mult( jacobian, dir, directionalDeriv );

		// go in the negative gradient direction to minimize cost
		CommonOps.scale( -1, dir );
	}

	public void computeDirection()
	{
		CommonOps.solve( jacobian, errorV, dir );

		double norm = NormOps.normP2( dir );
		CommonOps.divide( norm, dir );

		// compute the directional derivative
		CommonOps.mult( jacobian, dir, directionalDeriv );

		//
		CommonOps.multTransA( dir, directionalDeriv, descentDirectionMag );

		logger.debug( "descentDirectionMag: " + descentDirectionMag.get( 0 ) );
	}

	/**
	 * Uses Backtracking Line search to determine a step size.
	 * 
	 * @param c the armijoCondition parameter
	 * @param beta the fraction to multiply the step size at each iteration ( less than 1 )
	 * @param maxtries max number of tries
	 * @param t0 initial step size
	 * @return the step size
	 */
	public double backtrackingLineSearch( double c, double beta, int maxtries, double t0 )
	{
		double t = t0; // step size

		int k = 0;
		// boolean success = false;
		while ( k < maxtries )
		{
			if ( armijoCondition( c, t ) )
			{
				// success = true;
				break;
			}
			else
				t *= beta;

			k++;
		}

		logger.trace( "selected step size after " + k + " tries" );

		return t;
	}

	/**
	 * Returns true if the armijo condition is satisfied.
	 * 
	 * @param c the c parameter
	 * @param t the step size
	 * @return true if the step size satisfies the condition
	 */
	public boolean armijoCondition( double c, double t )
	{
		double[] d = dir.data;
		double[] x = estimate.data; // give a convenient name

		double[] x_ap = new double[ ndims ];
		for ( int i = 0; i < ndims; i++ )
			x_ap[ i ] = x[ i ] + t * d[ i ];

		// don't have to do this in here - this should be reused
		// double[] phix = xfm.apply( x );
		// TODO make sure estimateXfm is updated at the correct time
		double[] phix = estimateXfm.data;
		double[] phix_ap = xfm.apply( x_ap );

		double fx = squaredError( phix );
		double fx_ap = squaredError( phix_ap );

		// descentDirectionMag is a scalar
		// computeExpectedDescentReduction();
//		CommonOps.multTransA( dir, directionalDeriv, descentDirectionMag );
//		logger.debug( "descentDirectionMag: " + descentDirectionMag.get( 0 ) );

		double m = sumSquaredErrorsDeriv( this.target.data, phix ) * descentDirectionMag.get( 0 );

		logger.trace( "   f( x )     : " + fx );
		logger.trace( "   f( x + ap ): " + fx_ap );
//		logger.debug( "   p^T d      : " + descentDirectionMag.get( 0 ));
//		logger.debug( "   m          : " + m );
//		logger.debug( "   c * m * t  : " + c * t * m );
		logger.trace( "   f( x ) + c * m * t: " + ( fx + c * t * m ) );

		if ( fx_ap < fx + c * t * m )
			return true;
		else
			return false;
	}

	public double squaredError( double[] x )
	{
		double error = 0;
		for ( int i = 0; i < ndims; i++ )
			error += ( x[ i ] - this.target.get( i ) ) * ( x[ i ] - this.target.get( i ) );

		return error;
	}

	public void updateEstimate( double stepSize )
	{
		logger.trace( "step size: " + stepSize );
		logger.trace( "estimate:\n" + estimate );

		// go in the negative gradient direction to minimize cost
//		CommonOps.scale( -stepSize / norm, dir );
//		CommonOps.addEquals( estimate, dir );
		
		// dir should be pointing in the descent direction
		CommonOps.addEquals( estimate, stepSize, dir );

		logger.trace( "new estimate:\n" + estimate );
	}
	
	public void updateEstimateNormBased( double stepSize )
	{
		logger.debug( "step size: " + stepSize );
		logger.trace( "estimate:\n" + estimate );

		double norm = NormOps.normP2( dir );
		logger.debug( "norm: " + norm );

		// go in the negative gradient direction to minimize cost
		if ( norm > stepSize )
		{
			CommonOps.scale( -stepSize / norm, dir );
		}
		
		CommonOps.addEquals( estimate, dir );
		
		logger.trace( "new estimate:\n" + estimate );
	}

	public void updateError()
	{
		if ( estimate == null || target == null )
		{
			System.err.println( "WARNING: Call to updateError with null target or estimate" );
			return;
		}

		// errorV = estimate - target
//		CommonOps.sub( estimateXfm, target, errorV );
		
		// ( errorV = target - estimateXfm  )
		CommonOps.sub( target, estimateXfm, errorV );
		
		logger.trace( "#########################" );
		logger.trace( "updateError, estimate   :\n" + estimate );
		logger.trace( "updateError, estimateXfm:\n" + estimateXfm );
		logger.trace( "updateError, target     :\n" + target );
		logger.trace( "updateError, error      :\n" + errorV );
		logger.trace( "#########################" );
		
		// set scalar error equal to max of component-wise errors
		error = Math.abs( errorV.get( 0 ) );
		for ( int i = 1; i < ndims; i++ )
		{
			if ( Math.abs( errorV.get( i ) ) > error )
				error = Math.abs( errorV.get( i ) );
		}

	}

	/**
	 * This function returns \nabla f ^T \nabla f where f = || y - x ||^2 and
	 * the gradient is taken with respect to x
	 * 
	 * @param y
	 * @param x
	 * @return
	 */
	private double sumSquaredErrorsDeriv( double[] y, double[] x )
	{
		double errDeriv = 0.0;
		for ( int i = 0; i < ndims; i++ )
			errDeriv += ( y[ i ] - x[ i ] ) * ( y[ i ] - x[ i ] );

		return 2 * errDeriv;
	}

	public static double sumSquaredErrors( double[] y, double[] x )
	{
		int ndims = y.length;

		double err = 0.0;
		for ( int i = 0; i < ndims; i++ )
			err += ( y[ i ] - x[ i ] ) * ( y[ i ] - x[ i ] );

		return err;
	}

	public static void copyVectorIntoArray( DenseMatrix64F vec, double[] array )
	{
		System.arraycopy( vec.data, 0, array, 0, vec.getNumElements() );
	}

}

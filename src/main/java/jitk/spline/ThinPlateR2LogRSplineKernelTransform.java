package jitk.spline;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * Subclass of {@link KernelTransform} that implements a Thin-plate spline transformation.
 * Ported from itk's itkThinPlateSplineR2LogRKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms,
 * “A physics-based coordinate transformation for 3-D image matching.,”
 * IEEE Trans. Med. Imaging, vol. 16, no. 3, pp. 317–28, Jun. 1997.
 *
 * @author Kitware (ITK)
 * @author John Bogovic
 *
 */
public class ThinPlateR2LogRSplineKernelTransform extends KernelTransform {

	protected double eps = 1e-8;

	protected static Logger logger = LogManager.getLogger(ThinPlateR2LogRSplineKernelTransform.class.getName());

	public ThinPlateR2LogRSplineKernelTransform(){
		super();
	}
	public ThinPlateR2LogRSplineKernelTransform( final int ndims ){
		super( ndims );
	}
	public ThinPlateR2LogRSplineKernelTransform( final int ndims, final double[][] srcPts, final double[][] tgtPts)
	{
		super( ndims, srcPts, tgtPts );
	}
	public ThinPlateR2LogRSplineKernelTransform( final int ndims, final float[][] srcPts, final float[][] tgtPts)
	{
		super( ndims, srcPts, tgtPts );
	}
	public ThinPlateR2LogRSplineKernelTransform( final int ndims, final double[][] srcPts, final double[][] tgtPts, final double[] weights )
	{
		super( ndims, srcPts, tgtPts, weights );
	}

	public ThinPlateR2LogRSplineKernelTransform(
			final double[][] srcPts,
			final double[][] aMatrix,
			final double[] bVector,
			final double[] dMatrixData) {
		super(srcPts, aMatrix, bVector, dMatrixData);
	}

	@Override
	public void computeG(final double[] pt, final DenseMatrix64F mtx) {

		final double r = Math.sqrt(normSqrd(pt));
		final double nrm = r2Logr(r);

		CommonOps.setIdentity(mtx);
		CommonOps.scale(nrm,mtx);

	}

	@Override
	public void computeG(final double[] pt, final DenseMatrix64F mtx, final double w ) {

		computeG( pt, mtx );
		CommonOps.scale(w,mtx);

	}

	@Override
	public double[] computeDeformationContribution(final double[] thispt)
	{
		final double[] res = new double[ndims];
		final double[] diff = new double[ndims];

		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			srcPtDisplacement(lnd, thispt, diff);
			final double nrm = r2Logr( Math.sqrt(normSqrd(diff)) );

			for (int d = 0; d < ndims; d++) {
				res[d] += nrm * dMatrix.get(d, lnd);
			}
		}
		return res;
	}

	private double r2Logr( final double r ){
		double nrm = 0;
		if( r > eps){
			nrm = r * r * Math.log(r);
		}
		return nrm;
	}

}

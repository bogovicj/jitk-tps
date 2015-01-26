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
public class ThinPlateR2LogRSplineKernelTransformFloat extends KernelTransformFloat {

	static final protected double EPS = 1e-8;

	protected float[] diff;

	protected static Logger logger = LogManager.getLogger(ThinPlateR2LogRSplineKernelTransformFloat.class.getName());

	public ThinPlateR2LogRSplineKernelTransformFloat( final int ndims )
	{
		super( ndims );
		diff = new float[ndims];
	}

	public ThinPlateR2LogRSplineKernelTransformFloat( final int ndims, final float[][] srcPts, final float[][] tgtPts)
	{
		super( ndims, srcPts, tgtPts );
	}

   public ThinPlateR2LogRSplineKernelTransformFloat( final float[][] srcPts, final float[][] aMatrix, final float[] bVector, final double[] dMatrixData )
   {
      super( srcPts, aMatrix, bVector, dMatrixData);
   }

	@Override
	public void computeG(final float[] pt, final DenseMatrix64F mtx) {
		final double r = Math.sqrt(normSqrd(pt));
		final double nrm = r2Logr(r);

		CommonOps.setIdentity(mtx);
		CommonOps.scale(nrm,mtx);

	}

	@Override
	public void computeDeformationContribution(final float[] thispt) {

		for (int i = 0; i < ndims; ++i) {
			tmp[i] = 0;
			diff[i] = 0;
		}

		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			srcPtDisplacement(lnd, thispt, diff);
			final double nrm = r2Logr(Math.sqrt(normSqrd(diff)));

			for (int d = 0; d < ndims; d++) {
				tmp[d] += nrm * dMatrix.get(d, lnd);
			}
		}
	}

	private double r2Logr( final double r ){
		double nrm = 0;
		if( r > EPS){
			nrm = r * r * Math.log(r);
		}
		return nrm;
	}



}

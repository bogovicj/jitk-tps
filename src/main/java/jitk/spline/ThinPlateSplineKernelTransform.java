package jitk.spline;


import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;


/**
 * Subclass of {@link KernelTransform} that implements a Thin-plate spline
 * transformation. Ported from itk's itkThinPlateSplineKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms, “A physics-based
 * coordinate transformation for 3-D image matching.,” IEEE Trans. Med. Imaging,
 * vol. 16, no. 3, pp. 317–28, Jun. 1997.
 * 
 * @author Kitware (ITK)
 * @author John Bogovic
 * 
 */
public class ThinPlateSplineKernelTransform extends KernelTransform {

	protected static Logger logger = LogManager
			.getLogger(ThinPlateSplineKernelTransform.class.getName());

	public ThinPlateSplineKernelTransform() {
		super();
	}

	public ThinPlateSplineKernelTransform(int ndims, double[][] srcPts,
			double[][] tgtPts) {
		super(ndims, srcPts, tgtPts);
	}

	@Override
	public void computeG(double[] pt, DenseMatrix64F mtx) {

		double nrm = Math.sqrt(normSqrd(pt));

		CommonOps.setIdentity(mtx);
		CommonOps.scale(nrm, mtx);

	}

	@Override
	public void computeDeformationContribution(final double[] thispt,
			final double[] result) {

		for (int i = 0; i < ndims; ++i) {
			result[i] = 0;
			tmpDisplacement[i] = 0;
		}

		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			srcPtDisplacement(lnd, thispt, tmpDisplacement);
			double nrm = Math.sqrt(normSqrd(tmpDisplacement));

			for (int d = 0; d < ndims; d++) {
				result[d] += nrm * dMatrix.get(d, lnd);
			}
		}
	}

}

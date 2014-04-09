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
public class ThinPlateR2LogRSplineKernelTransformFloat extends KernelTransformFloatSeparable {

	protected double eps = 1e-8;

	protected static Logger logger = LogManager.getLogger(ThinPlateR2LogRSplineKernelTransformFloat.class.getName());

	public ThinPlateR2LogRSplineKernelTransformFloat(){
		super();
	}

	public ThinPlateR2LogRSplineKernelTransformFloat( int ndims, float[][] srcPts, float[][] tgtPts)
	{
		super( ndims, srcPts, tgtPts );
	}


     
   public ThinPlateR2LogRSplineKernelTransformFloat( float[][] srcPts, float[][] aMatrix, float[] bVector, double[] dMatrixData )
   {
      super( srcPts, aMatrix, bVector, dMatrixData);
   }

	@Override
	public double computeG( float[] pt ) {

		double r = Math.sqrt(normSqrd(pt));
		double nrm = r2Logr(r);
		return nrm;

	}

	@Override
	public float[] computeDeformationContribution(float[] thispt) 
	{
		float[] res = new float[ndims];
		float[] diff = new float[ndims];
		  
		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			srcPtDisplacement(lnd, thispt, diff); 
			double nrm = r2Logr( Math.sqrt(normSqrd(diff)) );

			for (int d = 0; d < ndims; d++) {
				res[d] += nrm * dMatrix.get(d, lnd);
			}
		}
		return res;
	}

	private double r2Logr( double r ){
		double nrm = 0;
		if( r > eps){
			nrm = r * r * Math.log(r);
		}
		return nrm;
	}

}

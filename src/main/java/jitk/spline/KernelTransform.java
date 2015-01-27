package jitk.spline;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.ops.CommonOps;

/**
 * Abstract superclass for kernel transform methods, for example,
 * {@link ThinPlateSplineKernelTransform}. Ported from itk's
 * itkKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms, “A physics-based
 * coordinate transformation for 3-D image matching.,” IEEE Trans. Med. Imaging,
 * vol. 16, no. 3, pp. 317–28, Jun. 1997.
 * 
 * @author Kitware (ITK)
 * @author John Bogovic
 * 
 */
public abstract class KernelTransform {

	protected int ndims;

	/* temporary vectors for intermediate calculations */
	protected double[] tmp;
	protected double[] tmpDisplacement;

	protected DenseMatrix64F gMatrix;
	protected DenseMatrix64F pMatrix;
	protected DenseMatrix64F kMatrix;
	protected DenseMatrix64F dMatrix;
	protected DenseMatrix64F wMatrix;
	protected DenseMatrix64F lMatrix;
	protected DenseMatrix64F yMatrix;

	protected DenseMatrix64F I;

	protected double[][] aMatrix;
	protected double[] bVector;

	protected double stiffness = 0.0; // reasonable values take the range [0.0,
										// 0.5]
	protected boolean wMatrixComputeD = false;
	protected boolean computeAffine = true;

	protected int nLandmarks;
	protected double[][] sourceLandmarks;
	protected double[][] targetLandmarks;
	protected double[] weights; // TODO: make the weights do something :-P

	protected double[][] displacement; // TODO: do we need this? yMatrix seems
										// to hold the same values

	// parameters relating
	protected int initialContainerSize = 100;
	protected double increaseRaio = 0.25;
	protected int containerSize;
	
	
	public static enum SolverType{ ADAPTIVE, PSEUDOINVERSE, LINEAR, LEAST_SQUARES, 
								   LEAST_SQUARES_QR, GENERAL };
	
	public SolverType solverType = SolverType.ADAPTIVE;
	
	protected static Logger logger = LogManager.getLogger(KernelTransform.class
			.getName());

	// TODO: Many of these methods could be optimized by performing them without
	// explicit construction / multiplication of the matrices.
	public KernelTransform() {
	}

	/*
	 * Constructor
	 */
	public KernelTransform(final int ndims) {
		// logger.info("initializing");

		this.ndims = ndims;

		tmp = new double[ndims];
		tmpDisplacement = new double[ndims];
		gMatrix = new DenseMatrix64F(ndims, ndims);

		/*
		 * identity matrix for convenience in building matrix during solving.
		 */
		I = new DenseMatrix64F(ndims, ndims);
		CommonOps.setIdentity(I);

		nLandmarks = 0;
		sourceLandmarks = new double[ndims][initialContainerSize];
		targetLandmarks = new double[ndims][initialContainerSize];
		containerSize = initialContainerSize;
	}

	/*
	 * Constructor with point matches
	 */
	public KernelTransform(final int ndims, final double[][] srcPts,
			final double[][] tgtPts) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

	/*
	 * Constructor with point matches
	 */
	public KernelTransform(final int ndims, final float[][] srcPts,
			final float[][] tgtPts) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

	/*
	 * Constructor with weighted point matches
	 */
	public KernelTransform(final int ndims, final double[][] srcPts,
			final double[][] tgtPts, final double[] weights) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
		setWeights(weights);
	}

	/**
	 * Constructor with transformation parameters. aMatrix and bVector are
	 * allowed to be null
	 */
	public KernelTransform(final double[][] srcPts, final double[][] aMatrix,
			final double[] bVector, final double[] dMatrixData) {

		this.ndims = srcPts.length;
		this.nLandmarks = srcPts[0].length;
		
		tmp = new double[ndims];
		tmpDisplacement = new double[ndims];
		gMatrix = new DenseMatrix64F(ndims, ndims);

		this.sourceLandmarks = srcPts;
		this.aMatrix = aMatrix;
		this.bVector = bVector;

		dMatrix = new DenseMatrix64F(ndims, nLandmarks);
		dMatrix.setData(dMatrixData);

	}

	public int getNumLandmarks() {
		return this.nLandmarks;
	}

	public int getNumDims() {
		return ndims;
	}

	public double[][] getSourceLandmarks() {
		return sourceLandmarks;
	}

	public double[][] getAffine() {
		return aMatrix;
	}

	public double[] getTranslation() {
		return bVector;
	}

	public double[] getKnotWeights() {
		return dMatrix.getData();
	}

	/*
	 * Sets the source and target landmarks for this KernelTransform object
	 * 
	 * @param sourcePts the collection of source points
	 * 
	 * @param targetPts the collection of target/destination points
	 */
	public void setLandmarks(final double[][] srcPts, final double[][] tgtPts)
			throws IllegalArgumentException {

		nLandmarks = srcPts[0].length;

		// check innput validity
		if (srcPts.length != ndims || tgtPts.length != ndims) {
			logger.error("Source and target landmark lists must have " + ndims
					+ " spatial dimentions.");
			return;
		}
		if (srcPts[0].length != nLandmarks || tgtPts[0].length != nLandmarks) {
			logger.error("Source and target landmark lists must be the same size");
			return;
		}

		this.sourceLandmarks = srcPts;
		this.targetLandmarks = tgtPts;

	}

	/*
	 * Sets the source and target landmarks for this KernelTransform object
	 * 
	 * @param sourcePts the collection of source points
	 * 
	 * @param targetPts the collection of target/destination points
	 */
	public void setLandmarks(final float[][] srcPts, final float[][] tgtPts)
			throws IllegalArgumentException {

		assert srcPts.length == ndims && tgtPts.length == ndims : "Source and target landmark lists must have "
				+ ndims + " spatial dimentions.";

		nLandmarks = srcPts[0].length;

		if (srcPts[0].length != nLandmarks || tgtPts[0].length != nLandmarks) {
			logger.error("Source and target landmark lists must be the same size");
			return;
		}

		sourceLandmarks = new double[ndims][nLandmarks];
		targetLandmarks = new double[ndims][nLandmarks];

		for (int d = 0; d < ndims; ++d) {
			for (int i = 0; i < nLandmarks; ++i) {
				sourceLandmarks[d][i] = srcPts[d][i];
				targetLandmarks[d][i] = tgtPts[d][i];
			}
		}
	}

	public void addMatch(final float[] source, final float[] target) {
		final double[] src = new double[source.length];
		final double[] tgt = new double[target.length];
		for (int i = 0; i < source.length; i++) {
			src[i] = source[i];
			tgt[i] = target[i];
		}
		addMatch(src, tgt);
	}

	public void addMatch(final double[] source, final double[] target) {
		if (nLandmarks + 1 > containerSize) {
			expandLandmarkContainers();
		}
		for (int d = 0; d < ndims; d++) {
			sourceLandmarks[d][nLandmarks] = source[d];
			targetLandmarks[d][nLandmarks] = target[d];
		}
		nLandmarks++;
	}

	protected void expandLandmarkContainers() {
		final int newSize = containerSize
				+ (int) Math.round(increaseRaio * containerSize);
		// logger.debug("increasing container size from " + containerSize +
		// " to " + newSize );
		final double[][] NEWsourceLandmarks = new double[ndims][newSize];
		final double[][] NEWtargetLandmarks = new double[ndims][newSize];

		for (int d = 0; d < ndims; d++)
			for (int n = 0; n < nLandmarks; n++) {
				NEWsourceLandmarks[d][n] = sourceLandmarks[d][n];
				NEWtargetLandmarks[d][n] = targetLandmarks[d][n];
			}

		containerSize = newSize;
		sourceLandmarks = NEWsourceLandmarks;
		targetLandmarks = NEWtargetLandmarks;
	}

	/**
	 * Sets the weights. Checks that the length matches the number of landmarks.
	 */
	private void setWeights(final double[] weights) {
		// make sure the length matches number
		// of landmarks
		if (weights == null) {
			return;
		}
		if (weights.length != this.nLandmarks) {
			this.weights = weights;
		} else {
			logger.error("weights have length (" + weights.length
					+ ") but tmust have length equal to number of landmarks "
					+ this.nLandmarks);
		}
	}

	public void setDoAffine(final boolean estimateAffine) {
		this.computeAffine = estimateAffine;
	}

	private void initMatrices() {

		dMatrix = new DenseMatrix64F(ndims, nLandmarks);
		kMatrix = new DenseMatrix64F(ndims * nLandmarks, ndims * nLandmarks);

		if (computeAffine) {
			aMatrix = new double[ndims][ndims];
			bVector = new double[ndims];

			pMatrix = new DenseMatrix64F((ndims * nLandmarks),
					(ndims * (ndims + 1)));
			lMatrix = new DenseMatrix64F(ndims * (nLandmarks + ndims + 1),
					ndims * (nLandmarks + ndims + 1));
			wMatrix = new DenseMatrix64F((ndims * nLandmarks) + ndims
					* (ndims + 1), 1);
			yMatrix = new DenseMatrix64F(ndims * (nLandmarks + ndims + 1), 1);
		} else {
			// we dont need the P matrix and L can point
			// directly to K rather than itself being initialized

			// the W matrix won't hold the affine component
			wMatrix = new DenseMatrix64F(ndims * nLandmarks, 1);
			yMatrix = new DenseMatrix64F(ndims * nLandmarks, 1);
		}

		displacement = new double[nLandmarks][ndims];
	}

	protected DenseMatrix64F computeReflexiveG() {
		CommonOps.fill(gMatrix, 0);
		for (int i = 0; i < ndims; i++) {
			gMatrix.set(i, i, stiffness);
		}
		return gMatrix;
	}

	protected void computeDeformationContribution(final double[] thispt) {
		computeDeformationContribution(thispt, tmp);
	}

	public void computeDeformationContribution(final double[] thispt,
			final double[] result) {

		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			computeG(result, gMatrix);

			for (int i = 0; i < ndims; i++) {
				for (int j = 0; j < ndims; j++) {
					result[j] += gMatrix.get(i, j) * dMatrix.get(i, lnd);
				}
			}
		}
	}

	protected void computeD() {
		for (int d = 0; d < ndims; d++)
			for (int i = 0; i < nLandmarks; i++) {
				displacement[i][d] = targetLandmarks[d][i]
						- sourceLandmarks[d][i];
			}
	}

	protected double normSqrd(final double[] v) {
		double nrm = 0;
		for (int i = 0; i < v.length; i++) {
			nrm += v[i] * v[i];
		}
		return nrm;
	}

	
	/**
	 * 
	 */
	public void solve(){
		computeW();
	}
	
	/**
	 * The main workhorse method.
	 * <p>
	 * Implements Equation (5) in Davis et al. and calls reorganizeW.
	 * 
	 */
	protected void computeW() {

		initMatrices();

		computeL();
		computeY();

		// TODO:during benchmarking, vary the solver

		final LinearSolver<DenseMatrix64F> solver;
		/*
		 * use pseudoinverse for underdetermined system linear solver otherwise
		 */
		if( solverType.equals( SolverType.ADAPTIVE )) {
			if (nLandmarks < ndims * ndims) {
				logger.debug("pseudo - inverse solver");
				solver = LinearSolverFactory.pseudoInverse(true);
			} else {
				logger.debug("linear solver");
				solver = LinearSolverFactory.linear(lMatrix.numCols);
			}
		}else{
			switch (solverType) {
			case LINEAR:
				logger.debug("Using LINEAR solver");
				solver = LinearSolverFactory.linear(lMatrix.numCols);
				break;
			case PSEUDOINVERSE:
				logger.debug("Using PSEUDOINVERSE solver");
				solver = LinearSolverFactory.pseudoInverse(true);
				break;	
			case LEAST_SQUARES:
				logger.debug("Using LEAST_SQUARES solver");
				solver = LinearSolverFactory.leastSquares(lMatrix.numRows, lMatrix.numCols);
				break;
			case LEAST_SQUARES_QR:
				logger.debug("Using LEAST_SQUARES_QR solver");
				solver = LinearSolverFactory.leastSquaresQrPivot(true, false);
				break;
			case GENERAL:
				logger.debug("Using GENERAL solver");
				solver = LinearSolverFactory.general(lMatrix.numRows, lMatrix.numCols);
				break;	
			default:
				logger.error("Invalid solver specified");
				solver = null;
				break;

			}
		}

		// solve linear system
		solver.setA(lMatrix);
		solver.solve(yMatrix, wMatrix);

		reorganizeW();
	}

	protected void computeL() {

		computeK();

		// fill P matrix if the affine parameters need to be computed
		if (computeAffine) {
			computeP();

			CommonOps.insert(kMatrix, lMatrix, 0, 0);
			CommonOps.insert(pMatrix, lMatrix, 0, kMatrix.getNumCols());
			CommonOps.transpose(pMatrix);

			CommonOps.insert(pMatrix, lMatrix, kMatrix.getNumRows(), 0);
			CommonOps.insert(kMatrix, lMatrix, 0, 0);
			// P matrix should be zero if points are already affinely aligned
			// bottom left O2 is already zeros after initializing 'lMatrix'
		} else {
			// in this case the L matrix
			// consists only of the K block.
			lMatrix = kMatrix;
		}

	}

	protected void computeP() {

		final DenseMatrix64F tmp = new DenseMatrix64F(ndims, ndims);

		for (int i = 0; i < nLandmarks; i++) {

			for (int d = 0; d < ndims; d++) {

				CommonOps.scale(sourceLandmarks[d][i], I, tmp);
				CommonOps.insert(tmp, pMatrix, i * ndims, d * ndims);

			}
			CommonOps.insert(I, pMatrix, i * ndims, ndims * ndims);
		}
	}

	/**
	 * Builds the K matrix from landmark points and G matrix.
	 */
	protected void computeK() {

		computeD();

		final double[] res = new double[ndims];

		for (int i = 0; i < nLandmarks; i++) {

			final DenseMatrix64F G = computeReflexiveG();
			CommonOps.insert(G, kMatrix, i * ndims, i * ndims);

			for (int j = i + 1; j < nLandmarks; j++) {

				srcPtDisplacement(i, j, res);
				computeG(res, G);

				CommonOps.insert(G, kMatrix, i * ndims, j * ndims);
				CommonOps.insert(G, kMatrix, j * ndims, i * ndims);
			}
		}
		// logger.debug(" kMatrix: \n" + kMatrix + "\n");
	}

	/**
	 * Fills the y matrix with the landmark point displacements.
	 */
	protected void computeY() {

		CommonOps.fill(yMatrix, 0);

		for (int i = 0; i < nLandmarks; i++) {
			for (int j = 0; j < ndims; j++) {
				yMatrix.set(i * ndims + j, 0, displacement[i][j]);
			}
		}
		if (computeAffine) {
			for (int i = 0; i < ndims * (ndims + 1); i++) {
				yMatrix.set(nLandmarks * ndims + i, 0, 0);
			}
		}

	}

	/**
	 * Copies data from the W matrix to the D, A, and b matrices which represent
	 * the deformable, affine and translational portions of the transformation,
	 * respectively.
	 */
	protected void reorganizeW() {

		int ci = 0;

		// the deformable (non-affine) part of the transform
		for (int lnd = 0; lnd < nLandmarks; lnd++) {
			for (int i = 0; i < ndims; i++) {
				dMatrix.set(i, lnd, wMatrix.get(ci, 0));
				ci++;
			}
		}
		// logger.debug(" dMatrix:\n" + dMatrix);

		if (computeAffine) {
			// the affine part of the transform
			for (int j = 0; j < ndims; j++)
				for (int i = 0; i < ndims; i++) {
					aMatrix[i][j] = wMatrix.get(ci, 0);
					ci++;
				}
			// logger.debug(" affine:\n" + XfmUtils.printArray(aMatrix));

			// the translation part of the transform
			for (int k = 0; k < ndims; k++) {
				bVector[k] = wMatrix.get(ci, 0);
				ci++;
			}
			// logger.debug(" b:\n" + XfmUtils.printArray(bVector) +"\n");
		}
		wMatrix = null;

	}

	/**
	 * Transforms the input point according to the affine part of the thin plate
	 * spline stored by this object.
	 * 
	 * @param pt
	 *            the point to be transformed
	 * @return the transformed point
	 */
	public double[] transformPointAffine(final double[] pt) {

		final double[] result = new double[ndims];
		// affine part
		for (int i = 0; i < ndims; i++) {
			for (int j = 0; j < ndims; j++) {
				result[i] += aMatrix[i][j] * pt[j];
			}
		}

		// translational part
		for (int i = 0; i < ndims; i++) {
			result[i] += bVector[i] + pt[i];
		}

		return result;
	}

	/**
	 * Transform a source vector pt into a target vector result.  pt and result
	 * must NOT be the same vector.
	 *
	 * @param pt
	 * @param result
	 */
	public void transform(final double[] pt, final double[] result) {

		computeDeformationContribution(pt, result);

		if (aMatrix != null) {
			// affine part
			for (int i = 0; i < ndims; i++)
				for (int j = 0; j < ndims; j++) {
					result[i] += aMatrix[i][j] * pt[j];
				}
		} else {
			for (int i = 0; i < ndims; i++) {
				result[i] += pt[i];
			}
		}

		if (bVector != null) {
			// translational part
			for (int i = 0; i < ndims; i++) {
				result[i] += bVector[i] + pt[i];
			}
		}
		/* TODO why is there no else? */
	}

	
	/**
	 * Transforms the input point according to the thin plate spline stored by
	 * this object.
	 * 
	 * @param pt
	 *            the point to be transformed
	 * @return the transformed point
	 */
	public double[] transform(final double[] pt) {
		final double[] result = new double[ndims];

		transform(pt, result);

		return result;
	}

	/**
	 * Transform pt in place.
	 * @param pt
	 */
	public void transformInPlace(final double[] pt) {

		transform(pt, tmp);

		for (int i = 0; i < ndims; ++i) {
			pt[i] = tmp[i];
		}
	}

	/**
	 * Computes the displacement between the i^th and j^th source points.
	 * 
	 * Stores the result in the input array 'res' Does not validate inputs.
	 */
	protected void srcPtDisplacement(final int i, final int j,
			final double[] res) {
		for (int d = 0; d < ndims; d++) {
			res[d] = sourceLandmarks[d][i] - sourceLandmarks[d][j];
		}
	}

	/**
	 * Computes the displacement between the i^th source point and the input
	 * point.
	 * 
	 * Stores the result in the input array 'res'. Does not validate inputs.
	 */
	protected void srcPtDisplacement(final int i, final double[] pt,
			final double[] res) {
		for (int d = 0; d < ndims; d++) {
			res[d] = sourceLandmarks[d][i] - pt[d];
		}
	}

	public abstract void computeG(double[] pt, DenseMatrix64F mtx);

	public void computeG(final double[] pt, final DenseMatrix64F mtx,
			final double w) {
		computeG(pt, mtx);
		CommonOps.scale(w, mtx);
	}

}

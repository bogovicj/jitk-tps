package jitk.spline;

import java.util.Arrays;

import mpicbg.models.CoordinateTransform;

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
public class ThinPlateR2LogRSplineKernelTransform implements
		CoordinateTransform {

	private static final long serialVersionUID = -972934724062617822L;
	
	protected int ndims;

	/* temporary vectors for intermediate calculations */
	protected double[] tmp;
	protected double[] tmpDisplacement;
	
	/* bounding box */
	protected double[] newBoxMin;
	protected double[] newBoxMax;
	
	
	// keeps track of landmark pairs that are in use
	protected boolean[] isPairActive;

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
	protected int nLandmarksActive;
	protected double[][] sourceLandmarks;
	protected double[][] targetLandmarks;
	protected double[] weights; // TODO: make the weights do something :-P

	protected double[][] displacement; // TODO: do we need this? yMatrix seems
										// to hold the same values

	// parameters relating
	protected int initialContainerSize = 100;
	protected double increaseRaio = 0.25;
	protected int containerSize;

	protected static final double EPS = 1e-8;

	protected static Logger logger = LogManager
			.getLogger(ThinPlateR2LogRSplineKernelTransform.class.getName());

	// TODO: Many of these methods could be optimized by performing them without
	// explicit construction / multiplication of the matrices.
	public ThinPlateR2LogRSplineKernelTransform() {}

	/*
	 * Constructor
	 */
	public ThinPlateR2LogRSplineKernelTransform(final int ndims) {
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
		nLandmarksActive = 0;
		sourceLandmarks = new double[ndims][initialContainerSize];
		targetLandmarks = new double[ndims][initialContainerSize];
		displacement = new double[initialContainerSize][ndims];
		isPairActive = new boolean[initialContainerSize];
		Arrays.fill( isPairActive, true );
		
		containerSize = initialContainerSize;
	}

	/*
	 * Constructor with point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform(final int ndims,
			final double[][] srcPts, final double[][] tgtPts) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

	/*
	 * Constructor with point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform(final int ndims,
			final float[][] srcPts, final float[][] tgtPts) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

	/*
	 * Constructor with weighted point matches
	 */
	public ThinPlateR2LogRSplineKernelTransform(final int ndims,
			final double[][] srcPts, final double[][] tgtPts,
			final double[] weights) {
		this(ndims);
		setLandmarks(srcPts, tgtPts);
		setWeights(weights);
	}

	/**
	 * Constructor with transformation parameters. aMatrix and bVector are
	 * allowed to be null
	 */
	public ThinPlateR2LogRSplineKernelTransform(final double[][] srcPts,
			final double[][] aMatrix, final double[] bVector,
			final double[] dMatrixData) {

		this.ndims = srcPts.length;
		this.nLandmarks = srcPts[0].length;
		nLandmarksActive = this.nLandmarks;

		tmp = new double[ndims];
		tmpDisplacement = new double[ndims];
		gMatrix = new DenseMatrix64F(ndims, ndims);

		this.sourceLandmarks = srcPts;
		this.aMatrix = aMatrix;
		this.bVector = bVector;

		dMatrix = new DenseMatrix64F(ndims, nLandmarks);
		dMatrix.setData(dMatrixData);

	}
	
	public ThinPlateR2LogRSplineKernelTransform deepCopy()
	{
		ThinPlateR2LogRSplineKernelTransform tps = new 
				ThinPlateR2LogRSplineKernelTransform( ndims, 
						XfmUtils.deepCopy(sourceLandmarks), targetLandmarks  );
		
		tps.aMatrix = XfmUtils.deepCopy( aMatrix );
		tps.bVector = Arrays.copyOf( this.bVector, this.bVector.length );
		tps.dMatrix = this.dMatrix.copy();
		tps.isPairActive = this.isPairActive.clone();
		
		tps.nLandmarks = this.nLandmarks;
		tps.nLandmarksActive = this.nLandmarksActive;
		
		return tps;
	}
	
	public ThinPlateR2LogRSplineKernelTransform deepCopy2()
	{
		return new ThinPlateR2LogRSplineKernelTransform( 
				XfmUtils.deepCopy( sourceLandmarks ), XfmUtils.deepCopy( aMatrix ), 
				bVector.clone(), dMatrix.data.clone() );
	}

	public int getNumLandmarks()
	{
		return this.nLandmarks;
	}
	
	public int getNumActiveLandmarks() 
	{
		return nLandmarksActive;
	}

	public int getNumDims()
	{
		return ndims;
	}

	public double[][] getSourceLandmarks()
	{
		return sourceLandmarks;
	}
	
	public double[][] getTargetLandmarks()
	{
		return targetLandmarks;
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
		nLandmarksActive = nLandmarks;
		containerSize = nLandmarks;
		
		assert srcPts.length == ndims && tgtPts.length == ndims : "Source and target landmark lists must have "
				+ ndims + " spatial dimentions.";
		assert srcPts[0].length == tgtPts[0].length : "Source and target landmark lists must have"
				+ "the same number of points.";

		this.sourceLandmarks = srcPts;
		this.targetLandmarks = tgtPts;
		computeD();
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
		assert srcPts[0].length == tgtPts[0].length : "Source and target landmark lists must have"
				+ "the same number of points.";

		nLandmarks = srcPts[0].length;
		nLandmarksActive = nLandmarks;
		
		sourceLandmarks = new double[ndims][nLandmarks];
		targetLandmarks = new double[ndims][nLandmarks];
		displacement = new double[nLandmarks][ndims];
		containerSize = nLandmarks;

		for (int d = 0; d < ndims; ++d) {
			for (int i = 0; i < nLandmarks; ++i) {
				sourceLandmarks[d][i] = srcPts[d][i];
				targetLandmarks[d][i] = tgtPts[d][i];
				displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];  
			}
		}
	}
	
	public boolean validateTransformPoints()
	{
		double[] validtmp = new double[ ndims ];
		double[] pt = new double[ ndims ];
		
		for( int i = 0; i < nLandmarksActive; i++ )
		{
			if( !isPairActive[ i ] )
				continue;
			
			for( int d = 0; d < ndims; d++ )
				pt[ d ] = sourceLandmarks[ d ][ i ];
			
			apply( pt, validtmp );
			for( int d = 0; d < ndims; d++ )
			{
				double diff = targetLandmarks[ d ][ i ] - validtmp[ d ]; 
				if ( diff > 0.1 || diff < -0.1 )
				{
					System.out.println( "error for landmark: " + i );
					System.out.println( "   pt : " + pt[0] + " " + pt[1] );
					System.out.println( "   res: " + validtmp[0] + " " + validtmp[1] );
					return false;
				}
			}
		}
		
		return true;
	}
	
	public void disableLandmarkPair( final int i )
	{
		if( isPairActive[ i ] )
		{
			isPairActive[ i ] = false;
			nLandmarksActive--;
		}
	}
	
	public void enableLandmarkPair( final int i )
	{
		if( !isPairActive[ i ] )
		{
			isPairActive[ i ] = true;
			nLandmarksActive++;
		}
	}
	
	public boolean isActive( final int i )
	{
		return isPairActive[ i ];
	}
	
	public void removePoint( final int i )
	{
		int addme = 0;
		for( int j = 0; j < nLandmarks; j++ )
		{
			if( j == i )
				addme++;
			
			if( j + addme < containerSize )
			{
				for (int d = 0; d < ndims; d++) 
				{
					sourceLandmarks[ d ][ j ] = sourceLandmarks[ d ][ j + addme ];
					targetLandmarks[ d ][ j ] = targetLandmarks[ d ][ j + addme ];
					displacement[ j ][ d ] = targetLandmarks[ d ][ j ] - sourceLandmarks[ d ][ j ];
					isPairActive[ j ]= isPairActive[ j + addme ];
				}
			}
		}
		nLandmarks--;
		
		if( isPairActive[i])
			nLandmarksActive--;
	}

	public void updateSourceLandmark(final int i, final double[] newSource) {
		for (int d = 0; d < ndims; d++) {
			sourceLandmarks[d][i] = newSource[d];
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];  
		}
	}

	public void updateTargetLandmark(final int i, final double[] newTarget) {
		for (int d = 0; d < ndims; d++) {
			targetLandmarks[d][i] = newTarget[d];
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];  
		}
	}

	public void updateSourceLandmark(final int i, final float[] newSource) {
		for (int d = 0; d < ndims; d++) {
			sourceLandmarks[d][i] = newSource[d];
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];  
		}
	}

	public void updateTargetLandmark(final int i, final float[] newTarget) {
		for (int d = 0; d < ndims; d++) {
			targetLandmarks[d][i] = newTarget[d];
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];  
		}
	}

	public void addMatch(final float[] source, final float[] target) {
		if (nLandmarks + 1 >= containerSize) {
			expandLandmarkContainers();
		}
		for (int d = 0; d < ndims; d++) {
			sourceLandmarks[d][nLandmarks] = source[d];
			targetLandmarks[d][nLandmarks] = target[d];
			displacement[nLandmarks][d] = targetLandmarks[d][nLandmarks] - sourceLandmarks[d][nLandmarks]; 
		}
		nLandmarks++;
		nLandmarksActive++;
	}

	public void addMatch(final double[] source, final double[] target) {
		if (nLandmarks + 1 >= containerSize) {
			expandLandmarkContainers();
		}
		for (int d = 0; d < ndims; d++) {
			sourceLandmarks[d][nLandmarks] = source[d];
			targetLandmarks[d][nLandmarks] = target[d];
			displacement[nLandmarks][d] = targetLandmarks[d][nLandmarks] - sourceLandmarks[d][nLandmarks];
		}
		nLandmarks++;
		nLandmarksActive++;
	}

	protected void expandLandmarkContainers() {
		final int newSize = containerSize
				+ (int) Math.round(increaseRaio * containerSize);
		
		 logger.debug("increasing container size from " + containerSize +
		 " to " + newSize );
		 
		final double[][] NEWsourceLandmarks = new double[ndims][newSize];
		final double[][] NEWtargetLandmarks = new double[ndims][newSize];
		final double[][] NEWdisplacement = new double[newSize][ndims];
		final boolean[]  NEWisPairActive = new boolean[newSize]; 
		Arrays.fill( NEWisPairActive, true );
		
		for (int d = 0; d < ndims; d++)
			for (int i = 0; i < nLandmarks; i++) {
				NEWsourceLandmarks[d][i] = sourceLandmarks[d][i];
				NEWtargetLandmarks[d][i] = targetLandmarks[d][i];
				NEWdisplacement[i][d] = NEWtargetLandmarks[d][i] - NEWsourceLandmarks[d][i];
				NEWisPairActive[i] = isPairActive[i];
			}

		containerSize = newSize;
		sourceLandmarks = NEWsourceLandmarks;
		targetLandmarks = NEWtargetLandmarks;
		displacement = NEWdisplacement;
		isPairActive = NEWisPairActive;
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

//		System.out.println( "nLandmarks: " + nLandmarks );
//		System.out.println( "nLandmarksActive: " + nLandmarksActive );
		dMatrix = new DenseMatrix64F( ndims, nLandmarksActive );
		kMatrix = new DenseMatrix64F( ndims * nLandmarksActive, ndims * nLandmarksActive );

		if (computeAffine)
		{
			aMatrix = new double[ ndims ][ ndims ];
			bVector = new double[ ndims ];

			pMatrix = new DenseMatrix64F( (ndims * nLandmarksActive),
					(ndims * (ndims + 1)));
			lMatrix = new DenseMatrix64F( ndims * (nLandmarksActive + ndims + 1),	
					ndims * (nLandmarksActive + ndims + 1));
			wMatrix = new DenseMatrix64F( (ndims * nLandmarksActive) + ndims * (ndims + 1), 1);
			yMatrix = new DenseMatrix64F( ndims * (nLandmarksActive + ndims + 1), 1);
		}
		else 
		{
			// we dont need the P matrix and L can point
			// directly to K rather than itself being initialized

			// the W matrix won't hold the affine component
			wMatrix = new DenseMatrix64F( ndims * nLandmarksActive, 1 );
			yMatrix = new DenseMatrix64F( ndims * nLandmarksActive, 1 );
		}

	}

	protected DenseMatrix64F computeReflexiveG() {
		CommonOps.fill(gMatrix, 0);
		for (int i = 0; i < ndims; i++) {
			gMatrix.set(i, i, stiffness);
		}
		return gMatrix;
	}

	protected synchronized void computeDeformationContribution(final double[] thispt) {
		computeDeformationContribution(thispt, tmp);
	}

	protected void computeD() 
	{
		displacement = new double[nLandmarks][ndims];
		for (int d = 0; d < ndims; d++)
			for (int i = 0; i < nLandmarks; i++) 
			{
				displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];
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
	public void solve() {
		computeW();
	}

	/**
	 * The main workhorse method.
	 * <p>
	 * Implements Equation (5) in Davis et al. and calls reorganizeW.
	 *
	 */
	protected void computeW() {

//		System.out.println( "activePairs: " + XfmUtils.printArray( isPairActive ) );
		initMatrices();

		computeL();
		computeY();

		final LinearSolver<DenseMatrix64F> solver;
		// logger.debug("linear solver");
		if( nLandmarksActive < ndims*ndims )
		{
			logger.info("pseudo inverse solver");
			solver =  LinearSolverFactory.pseudoInverse(false);
		}else
		{
			logger.info("linear solver");
			solver =  LinearSolverFactory.linear(lMatrix.numCols);
		}
		
		// solve linear system
		solver.setA(lMatrix);
		solver.solve(yMatrix, wMatrix);

		reorganizeW();
		
	}

	protected void computeL() {
		
		// computeD();
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

	protected void computeP()
	{
		final DenseMatrix64F tmp = new DenseMatrix64F(ndims, ndims);

		int i = 0;
		int gi = 0;
		while( i < nLandmarks )
		{
			if( !isPairActive[i])
			{
				i++;
				continue;
			}
			for (int d = 0; d < ndims; d++)
			{
				CommonOps.scale( sourceLandmarks[d][i], I, tmp );
				CommonOps.insert( tmp, pMatrix, gi * ndims, d * ndims );
			}
			CommonOps.insert( I, pMatrix, gi * ndims, ndims * ndims );
			i++;
			gi++;
		}
	}

	/**
	 * Builds the K matrix from landmark points and G matrix.
	 */
	protected void computeK()
	{
		final double[] res = new double[ ndims ];

		int i = 0;
		int gi = 0;
		while( i < nLandmarks )
		{
			if( !isPairActive[i] ){
				i++;
				continue;
			}

			final DenseMatrix64F G = computeReflexiveG();
			CommonOps.insert( G, kMatrix, gi * ndims, gi * ndims );

			int j  = i + 1;
			int gj = gi + 1;
			while( j <  nLandmarks )
			{
				if( !isPairActive[j] ){
					j++;
					continue;
				}
				srcPtDisplacement( i, j, res );
				computeG( res, G );

				CommonOps.insert( G, kMatrix, gi * ndims, gj * ndims );
				CommonOps.insert( G, kMatrix, gj * ndims, gi * ndims );
				
				j++;
				gj++;
			}
			
			i++;
			gi++;
		}
	}

	/**
	 * Fills the y matrix with the landmark point displacements.
	 */
	protected void computeY()
	{
		CommonOps.fill(yMatrix, 0);

//		for (int i = 0; i < nLandmarks; i++) {
		int i = 0;
		int gi = 0;
		while( i < nLandmarks )
		{
			if( !isPairActive[i]){
				i++;
				continue;
			}
			for (int j = 0; j < ndims; j++) {
				yMatrix.set( gi * ndims + j, 0, displacement[i][j]);
			}
			i++;
			gi++;
		}
		if (computeAffine) {
			for ( i = 0; i < ndims * (ndims + 1); i++) {
				yMatrix.set( nLandmarksActive * ndims + i, 0, 0);
			}
		}
	}

	/**
	 * Copies data from the W matrix to the D, A, and b matrices which represent
	 * the deformable, affine and translational portions of the transformation,
	 * respectively.
	 */
	protected void reorganizeW() 
	{
		// the deformable (non-affine) part of the transform
		int ci = 0;
		int i = 0;
		int gi = 0;
		while( i < nLandmarks )
		{
			if( !isPairActive[i]){
				i++;
				continue;
			}
			for (int d = 0; d < ndims; d++) {
				dMatrix.set( d, gi, wMatrix.get(ci, 0));
				ci++;
			}
			i++;
			gi++;
		}

		// the affine part of the transform
		if (computeAffine) {
			// the affine part of the transform
			for (int j = 0; j < ndims; j++)
				for ( i = 0; i < ndims; i++) {
					aMatrix[i][j] = wMatrix.get(ci, 0);
					ci++;
				}

			// the translation part of the transform
			for (int k = 0; k < ndims; k++) {
				bVector[k] = wMatrix.get(ci, 0);
				ci++;
			}
		}
		
		wMatrix = null;
		
	}

	public void computeG(final double[] pt, final DenseMatrix64F mtx) {

		final double r = Math.sqrt(normSqrd(pt));
		final double nrm = r2Logr(r);

		CommonOps.setIdentity(mtx);
		CommonOps.scale(nrm, mtx);
	}

	public void computeG(final double[] pt, final DenseMatrix64F mtx,
			final double w) {

		computeG(pt, mtx);
		CommonOps.scale(w, mtx);
	}

	public synchronized void computeDeformationContribution(final double[] thispt,
			final double[] result) {

		for ( int i = 0; i < ndims; ++i ) 
		{
			result[i] = 0;
			tmpDisplacement[i] = 0;
		}

		int di = 0;
		for (int lnd = 0; lnd < nLandmarks; lnd++) 
		{
			if( isPairActive[ lnd ])
			{
				srcPtDisplacement( lnd, thispt, tmpDisplacement );
				final double nrm = r2Logr( Math.sqrt( normSqrd( tmpDisplacement )));
				
				for (int d = 0; d < ndims; d++) 
					result[d] += nrm * dMatrix.get( d, di );
			
				di++;
			}
		}
	}
	
	public void printXfmBacks2d( int maxx, int maxy, int delx, int dely )
	{
		double[] pt = new double[ 2 ];
		double[] result = new double[ 2 ];
		for( int x = 0; x < maxx; x+=delx )for( int y = 0; y < maxy; y+=dely )
		{
			pt[ 0 ] = x;
			pt[ 1 ] = y;
			
			this.apply( pt, result );
			System.out.println("( " + x + ", " + y + " )  ->  ( " + result[0] + ", " + result[0] + " )");
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
	
	public void apply(final double[] pt, final double[] result) 
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
	public void apply(final double[] pt, final double[] result, boolean debug ) 
	{
		if( debug )
		{
			System.out.println("nLandmarks " + nLandmarks );
			System.out.println("nLandmarksActive " + nLandmarksActive );
			System.out.println("dMatrix " + dMatrix.numRows + " " + dMatrix.numCols );
			System.out.println("isPairActive " + XfmUtils.printArray( isPairActive ) );
		}
		
		computeDeformationContribution( pt, result );

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
	@Override
	public double[] apply(final double[] pt) {
		final double[] result = new double[ndims];

		apply(pt, result);

		return result;
	}

	/**
	 * Transform pt in place.
	 *
	 * @param pt
	 */
	@Override
	public synchronized void applyInPlace(final double[] pt) {

		apply(pt, tmp);

		for (int i = 0; i < ndims; ++i) {
			pt[i] = tmp[i];
		}
	}
	
	/**
	 * Estimates the bounding box of this transformation.
	 * Stores the results in local variables that are accessible via get methods.
	 * @param min input minimum
	 * @param max input maximum
	 */
	public void estimateBoundingBox( double[] min, double[] max )
	{
		newBoxMin = new double[ ndims ];
		newBoxMax = new double[ ndims ];
		
		
	}
	
	public double[] getBoxMin()
	{
		return newBoxMin;
	}

	public double[] getBoxMax()
	{
		return newBoxMax;
	}
	
	/**
	 * Computes the displacement between the i^th and j^th source points.
	 *
	 * Stores the result in the input array 'res' Does not validate inputs.
	 */
	protected void srcPtDisplacement(final int i, final int j,
			final double[] res) {
		for (int d = 0; d < ndims; d++) 
		{
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

	private static double r2Logr(final double r) {
		double nrm = 0;
		if (r > EPS) {
			nrm = r * r * Math.log(r);
		}
		return nrm;
	}

	
}

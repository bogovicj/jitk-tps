package jitk.spline;

import mpicbg.models.CoordinateTransform;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.ops.CommonOps;


/**
 * Abstract superclass for kernel transform methods,
 * for example, {@link ThinPlateSplineKernelTransform}.
 * Ported from itk's itkKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms,
 * “A physics-based coordinate transformation for 3-D image matching.,”
 * IEEE Trans. Med. Imaging, vol. 16, no. 3, pp. 317–28, Jun. 1997.
 *
 * @author Kitware (ITK)
 * @author John Bogovic
 *
 */
public abstract class KernelTransformFloat implements CoordinateTransform {

    protected int ndims;

    /* temporary vector for doing calculations */
    protected float[] tmp;
    protected float[] tmp1;

	protected DenseMatrix64F gMatrix;
	protected DenseMatrix64F dMatrix;
	protected DenseMatrix64F wMatrix;
	protected DenseMatrix64F lMatrix;
	protected DenseMatrix64F yMatrix;

	protected DenseMatrix64F I;

	protected float[][] aMatrix;
	protected float[] bVector;

	protected double 	stiffness = 0.0; // reasonable values take the range [0.0, 0.5]
	protected boolean	wMatrixComputeD = false;
	protected boolean	computeAffine   = true;
	protected boolean	isInverse 	    = false;

	protected int 		   nLandmarks;
	protected float[][]    sourceLandmarks;
	protected float[][]    targetLandmarks;
	protected float[] 	   weights;  // TODO: make the weights do something :-P

	protected float[][] displacement; // TODO: do we need this? yMatrix seems to hold the same values

	//
	KernelTransformFloat inv = null;

	// parameters relating
	protected int 	 initialContainerSize = 100;
	protected double increaseRaio = 0.25;
	protected int 	 containerSize;

	protected static Logger logger = LogManager.getLogger(KernelTransformFloat.class.getName());

	//TODO: Many of these methods could be optimized by performing them without
	// explicit construction / multiplication of the matrices.
	public KernelTransformFloat(){}

   /*
    * Constructor
    */
	public KernelTransformFloat(final int ndims){
		//logger.info("initializing");

		this.ndims = ndims;
		tmp = new float[ndims];

		gMatrix = new DenseMatrix64F(ndims, ndims);

		I = new DenseMatrix64F(ndims, ndims);
		for (int i=0; i<ndims; i++){
			I.set(i,i,1);
		}

		nLandmarks = 0;
		sourceLandmarks = new float[ndims][initialContainerSize];
		targetLandmarks = new float[ndims][initialContainerSize];
		containerSize = initialContainerSize;

	}

	/*
	 * Constructor with point matches
	 */
	public KernelTransformFloat( final int ndims, final float[][] srcPts, final float[][] tgtPts){
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

	/*
	 * Constructor with point matches and weights
	 */
	public KernelTransformFloat( final int ndims, final float[][] srcPts, final float[][] tgtPts, final float[] weights ){
		this(ndims);
		setLandmarks(srcPts, tgtPts);
        setWeights( weights );
	}

	/**
	 * Constructor with transformation parameters. aMatrix and bVector are
	 * allowed to be null
	 */
	public KernelTransformFloat(final float[][] srcPts,
			final float[][] aMatrix, final float[] bVector,
			final double[] dMatrixData) {
		this.ndims = srcPts.length;
		tmp = new float[ndims];
		this.nLandmarks = srcPts[0].length;

		gMatrix = new DenseMatrix64F(ndims, ndims);

		this.sourceLandmarks = srcPts;
		this.aMatrix = aMatrix;
		this.bVector = bVector;

		dMatrix = new DenseMatrix64F(ndims, nLandmarks);
		dMatrix.setData(dMatrixData);
	}

   public int getNumLandmarks(){
      return this.nLandmarks;
   }

   public int getNumDims(){
      return ndims;
   }

   public float[][] getSourceLandmarks(){
      return sourceLandmarks;
   }

   public float[][] getTargetLandmarks(){
	   return targetLandmarks;
   }

   public float[][] getAffine(){
      return aMatrix;
   }

   public float[] getTranslation(){
      return bVector;
   }

   public double[] getKnotWeights(){
      return dMatrix.getData();
   }

   /*
    * Sets the source and target landmarks for this KernelTransform object
    *
    * @param sourcePts the collection of source points
    * @param targetPts the collection of target/destination points
    */
	public void setLandmarks( final float[][] srcPts, final float[][] tgtPts) throws IllegalArgumentException{

		nLandmarks = srcPts[0].length;

		// check innput validity
		if( srcPts.length != ndims ||
			 tgtPts.length != ndims )
		{
			logger.error("Source and target landmark lists must have " + ndims + " spatial dimentions.");
			return;
		}
		if(
				srcPts[0].length != nLandmarks ||
				tgtPts[0].length != nLandmarks ||
				tgtPts[1].length != nLandmarks )
		{
			logger.error("Source and target landmark lists must be the same size");
			return;
		}

		this.sourceLandmarks = srcPts;
		this.targetLandmarks = tgtPts;


		// consider calling computeW() here.
        // No - the work should be an explicit call, but
        // consider naming it something better

	}

	public void updateSourceLandmark( final int i, final float[] newSource ){
		for(int j=0; j<ndims; j++)
		{
			sourceLandmarks[j][i] = newSource[j];
		}
	}

	public void updateTargetLandmark( final int i, final float[] newTarget ){
		for(int j=0; j<ndims; j++)
		{
			targetLandmarks[j][i] = newTarget[j];
		}
	}

	public void addMatch( final double[] source, final double[] target )
	{
		final float[] src = new float[source.length];
		final float[] tgt = new float[target.length];
		for(int i=0; i<source.length; i++){
			src[i]= (float) source[i];
			tgt[i]= (float) target[i];
		}
		addMatch( src, tgt );
	}

	public void addMatch( final float[] source, final float[] target )
	{
		if( nLandmarks + 1 > containerSize ){
			expandLandmarkContainers();
		}
		for( int d = 0; d<ndims; d++){
			sourceLandmarks[d][nLandmarks] = source[d];
			targetLandmarks[d][nLandmarks] = target[d];
		}
		nLandmarks++;
	}

	protected void expandLandmarkContainers()
	{
		final int newSize = containerSize + (int) Math.round( increaseRaio * containerSize );
		//logger.debug("increasing container size from " + containerSize  + " to " + newSize );
		final float[][] NEWsourceLandmarks = new float[ndims][newSize];
		final float[][] NEWtargetLandmarks = new float[ndims][newSize];

		for( int d = 0; d<ndims; d++) for( int n = 0; n<nLandmarks; n++){
			NEWsourceLandmarks[d][n] = sourceLandmarks[d][n];
			NEWtargetLandmarks[d][n] = targetLandmarks[d][n];
		}

		containerSize   = newSize;
		sourceLandmarks = NEWsourceLandmarks;
		targetLandmarks = NEWtargetLandmarks;
	}

    /**
     * Sets the weights.  Checks that the length matches
     * the number of landmarks.
     */
   private void setWeights( final float[] weights ){
        // make sure the length matches number
        // of landmarks
        if( weights==null){
            return;
        }
        if( weights.length != this.nLandmarks ){
            this.weights = weights;
        }else{
            logger.error( "weights have length (" + weights.length  +
                    ") but tmust have length equal to number of landmarks " +
                    this.nLandmarks );
        }
   }

   public void setDoAffine(final boolean estimateAffine)
   {
      this.computeAffine = estimateAffine;
   }


   	private void initMatrices() {

		// pMatrix = new DenseMatrix64F( (ndims * nLandmarks), ( ndims * (ndims
		// + 1)) );
		// kMatrix = new DenseMatrix64F( ndims * nLandmarks, ndims *
		// nLandmarks);

		lMatrix = new DenseMatrix64F(ndims * (nLandmarks + ndims + 1), ndims
				* (nLandmarks + ndims + 1));

		yMatrix = new DenseMatrix64F(ndims * (nLandmarks + ndims + 1), 1);
		wMatrix = new DenseMatrix64F(
				(ndims * nLandmarks) + ndims * (ndims + 1), 1);
		dMatrix = new DenseMatrix64F(ndims, nLandmarks);

		if (computeAffine) {
			aMatrix = new float[ndims][ndims];
			bVector = new float[ndims];
		}

		displacement = new float[nLandmarks][ndims];
	}

	protected DenseMatrix64F computeReflexiveG(){
		CommonOps.fill(gMatrix, 0);
		for (int i=0; i<ndims; i++){
			gMatrix.set(i,i, stiffness);
		}
		return gMatrix;
	}

	protected void computeDeformationContribution( final float[] thispt ){
		computeDeformationContribution( thispt, tmp );
	}

	public void computeDeformationContribution(
			final float[] thispt,
			final float[] result) {

		// TODO: check for bugs - is l1 ever used?
		// double[] l1 = null;

		// logger.debug("dMatrix: " + dMatrix);

		for (int lnd = 0; lnd < nLandmarks; lnd++) {

			computeG(result, gMatrix);

			for (int i = 0; i < ndims; i++)
				for (int j = 0; j < ndims; j++) {
					result[j] += gMatrix.get(i, j) * dMatrix.get(i, lnd);
				}
		}
	}

	protected void computeD(){
		for( int d=0; d<ndims; d++ ) for( int i=0; i<nLandmarks; i++ ){
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i];
		}
	}

	protected float normSqrd( final float[] v ){
     float nrm = 0;
      for(int i=0; i<v.length; i++){
         nrm += v[i]*v[i];
      }
      return nrm;
   }


	public void solve(){
		computeW();
	}

	public void solve( final boolean isInverse ){
		this.isInverse = isInverse;
		computeW();
	}

	/**
	 * The main workhorse method.
	 * <p>
	 * Implements Equation (5) in Davis et al.
	 * and calls reorganizeW.
	 *
	 */
	public void computeW(){

		initMatrices();

		computeL();
		computeY();

		//logger.debug(" lMatrix: " + lMatrix);
		//logger.debug(" yMatrix: " + yMatrix);

		// solve linear system
		LinearSolver<DenseMatrix64F> solver = null;

		// use pseudoinverse for underdetermined system
		// linear solver otherwise
		if( nLandmarks < ndims*ndims )
		{
			//logger.debug("pseudo - inverse solver");
			solver =  LinearSolverFactory.pseudoInverse(true);
		}else
		{
			//logger.debug("linear solver");
			solver =  LinearSolverFactory.linear(lMatrix.numCols);
		}

//		LinearSolverFactory.general(lMatrix.numRows, lMatrix.numCols);

		solver.setA(lMatrix);
		solver.solve(yMatrix, wMatrix);

		//logger.debug("wMatrix:\n" + wMatrix );

		reorganizeW();

	}


	protected void computeL() {

		// fill P matrix if the affine parameters need to be computed
		if (computeAffine) {
			computeP();
		}
		// P matrix should be zero if points are already affinely aligned

		computeK();

		// bottom left O2 is already zeros after initializing 'lMatrix'
	}

	/**
	 * Inserts the blocks of the P matrix directly into the L matrix
	 */
	protected void computeP(){

		final int offset = ndims*nLandmarks;

		final DenseMatrix64F tmp = new DenseMatrix64F(ndims,ndims);

		for( int i=0; i<nLandmarks; i++ ){
			for( int d=0; d<ndims; d++ ){

				CommonOps.scale( sourceLandmarks[d][i], I, tmp);
				CommonOps.insert( tmp, lMatrix,  offset + d*ndims, i*ndims ); // maybe ok
				CommonOps.insert( tmp, lMatrix,  i*ndims, offset + d*ndims ); // good

			}
			CommonOps.insert( I, lMatrix,  offset + ndims*ndims, i*ndims ); // maybe ok
			CommonOps.insert( I, lMatrix,  i*ndims, offset + ndims*ndims ); // good
		}
	}


	/**
	 * Builds the K matrix from landmark points and G matrix
	 * but drops the results directly into the L matrix.
	 */
	protected void computeK(){

		computeD();

		final float[] res = new float[ndims];

		for( int i=0; i<nLandmarks; i++ ){

			final DenseMatrix64F G = computeReflexiveG();
			CommonOps.insert(G, lMatrix, i * ndims, i * ndims);

			for( int j = i+1; j<nLandmarks; j++ ){

				srcPtDisplacement(i,j,res);
				computeG(res, G);

				CommonOps.insert(G, lMatrix, i * ndims, j * ndims);
				CommonOps.insert(G, lMatrix, j * ndims, i * ndims);
			}
		}
		//logger.debug(" kMatrix: \n" + lMatrix + "\n");
	}


	/**
	 * Fills the y matrix with the landmark point displacements.
	 */
	protected void computeY(){

		CommonOps.fill( yMatrix, 0 );

		for (int i=0; i<nLandmarks; i++) {
			for (int j=0; j<ndims; j++) {
				yMatrix.set( i*ndims + j, 0, displacement[i][j]);
			}
		}
		for (int i=0; i< ndims * (ndims + 1); i++) {
			yMatrix.set( nLandmarks * ndims + i, 0, 0);
		}

	}

	/**
	 * Copies data from the W matrix to the D, A, and b matrices
	 * which represent the deformable, affine and translational
	 * portions of the transformation, respectively.
	 */
	protected void reorganizeW(){

		int ci = 0;

		// the deformable (non-affine) part of the transform
		for( int lnd=0; lnd<nLandmarks; lnd++){
			for (int i=0; i<ndims; i++) {
				dMatrix.set(i, lnd, wMatrix.get(ci, 0));
				ci++;
			}
		}
		//logger.debug(" dMatrix:\n" + dMatrix);

      if( computeAffine )
      {
         // the affine part of the transform
         for( int j=0; j<ndims; j++) for (int i=0; i<ndims; i++) {
            aMatrix[i][j] =  (float)wMatrix.get(ci,0);
            ci++;
         }
         //logger.debug(" affine:\n" + XfmUtils.printArray(aMatrix));

         // the translation part of the transform
         for( int k=0; k<ndims; k++) {
            bVector[k] = (float)wMatrix.get(ci, 0);
            ci++;
         }
		   //logger.debug(" b:\n" + XfmUtils.printArray(bVector) +"\n");
      }

      	//logger.debug(" ");

		wMatrix = null;
		yMatrix = null;
		lMatrix = null;
		System.gc();
	}

   /**
    * Transforms the input point according to the affine part of
    * the thin plate spline stored by this object.
    *
    * @param pt the point to be transformed
    * @return the transformed point
    */
	public double[] transformPointAffine(final double[] pt){

      final double[] result = new double[ndims];
      // affine part
		for(int i=0; i<ndims; i++){
			for(int j=0; j<ndims; j++){
            result[i] += aMatrix[i][j] * pt[j];
         }
      }

      // translational part
      for(int i=0; i<ndims; i++){
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
	public void transform(final float[] pt, final float[] result) {

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
	public float[] transformPoint(final float[] pt) {

		// logger.trace("transforming pt:  " + XfmUtils.printArray(pt));
		final float[] result = new float[ndims];

		transform(pt, result);

		return result;
	}

	/**
	 * Transform pt in place.
	 */
	public void transformInPlace(final float[] pt) {

		computeDeformationContribution(pt, tmp);

		for (int i = 0; i < ndims; ++i) {
			pt[i] = tmp[i];
		}
	}

	@Override
    public float[] apply(final float[] location) {
		return transformPoint( location );
	}

	@Override
    public void applyInPlace(final float[] location) {
		transformInPlace( location );
	}

	/**
	 * Computes the displacement between the i^th and j^th source points.
	 *
	 * Stores the result in the input array 'res'
	 * Does not validate inputs.
	 */
	protected void srcPtDisplacement(final int i, final int j, final float[] res)
	{
		for( int d=0; d<ndims; d++ ){
			res[d] = sourceLandmarks[d][i] - sourceLandmarks[d][j];
		}
	}

	/**
	 * Computes the displacement between the i^th source point
	 * and the input point.
	 *
	 * Stores the result in the input array 'res'.
	 * Does not validate inputs.
	 */
	protected void srcPtDisplacement(final int i, final float[] pt, final float[] res)
	{
		for( int d=0; d<ndims; d++ ){
			res[d] = sourceLandmarks[d][i] - pt[d];
		}
	}

	//protected float[] subtract(float[] p1, float[] p2){
	//	int nd = p1.length;
	//	float[] out = new float[nd];
	//	for (int d=0; d<nd; d++){
	//		out[d] = p1[d] - p2[d];
	//	}
	//	return out;
	//}
	//
	//protected float[] subtract(float[] p1, float[] p2, float[] out){
	//	int nd = out.length;
	//	for (int d=0; d<nd; d++){
	//		out[d] = p1[d] - p2[d];
	//	}
	//	return out;
	//}

	public abstract void computeG( float[] pt, DenseMatrix64F mtx);


}

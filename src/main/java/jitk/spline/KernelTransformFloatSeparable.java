package jitk.spline;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.ejml.data.*;
import org.ejml.factory.LinearSolver;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.ops.CommonOps;


/**
 * Abstract superclass for kernel transform methods,
 * for example, {@link ThinPlateSplineKernelTransform}.
 * Ported from itk's itkKernelTransform.hxx
 * <p>
 * M. H. Davis, a Khotanzad, D. P. Flamig, and S. E. Harms, 
 * ‚A physics-based coordinate transformation for 3-D image matching.,
 * IEEE Trans. Med. Imaging, vol. 16, no. 3, pp. 317‚28, Jun. 1997. 
 * 
 * The process() method is the correct 
 *  
 * ktfs = new 'subclassConstructor;
 * kfts.setDoAffine(doAffine);
 * kfts.process();
 * 
 *
 * @author Kitware (ITK)
 * @author John Bogovic
 *
 */
public abstract class KernelTransformFloatSeparable {
	
   protected int ndims;

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
	
	protected int 		      nLandmarks;
	protected float[][]    sourceLandmarks;
	protected float[][]    targetLandmarks;
	protected float[] 	   weights;  // TODO: make the weights do something :-P

	protected float[][] displacement; // TODO: do we need this? yMatrix seems to hold the same values
	
	protected static Logger logger = LogManager.getLogger(KernelTransformFloatSeparable.class.getName());
	
	//TODO: Many of these methods could be optimized by performing them without
	// explicit construction / multiplication of the matrices. 
	public KernelTransformFloatSeparable(){}

   /*
    * Constructor
    */
	public KernelTransformFloatSeparable(int ndims){
		logger.info("initializing");
		
		this.ndims = ndims;

		gMatrix = new DenseMatrix64F(ndims, ndims);

		I       = new DenseMatrix64F(ndims, ndims);
		for (int i=0; i<ndims; i++){
			I.set(i,i,1);
		}
		
	}

	/*
	 * Constructor with point matches 
	 */
	public KernelTransformFloatSeparable( int ndims, float[][] srcPts, float[][] tgtPts){
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

   /**
    * Constructor with transformation parameters.
    * aMatrix and bVector are allowed to be null
    */
   public KernelTransformFloatSeparable( float[][] srcPts, float[][] aMatrix, float[] bVector, double[] dMatrixData )
   {

      this.ndims = srcPts.length;
      this.nLandmarks = srcPts[0].length;

      this.sourceLandmarks = srcPts;
      this.aMatrix = aMatrix;
      this.bVector = bVector;

      dMatrix = new DenseMatrix64F( ndims, nLandmarks);
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
	public void setLandmarks( float[][] srcPts, float[][] tgtPts) throws IllegalArgumentException{

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
	
		//TODO consider calling computeW() here.
		
	}

   public void setDoAffine(boolean estimateAffine)
   { 
      this.computeAffine = estimateAffine; 
   } 
   
   /**
    * This method computes the transformation after source and target points have been set.
    */
   public void fit(){
	   if(computeAffine){
		   computeAffine();
		   updateDisplacementPostAffine();
	   }
		computePostAffineDef();
   }

   private void initMatricesDef()
	{
		
		lMatrix = new DenseMatrix64F( nLandmarks, nLandmarks );
		yMatrix = new DenseMatrix64F( nLandmarks, 1 );
		wMatrix = new DenseMatrix64F( nLandmarks, 1 );
		
		dMatrix = new DenseMatrix64F( ndims, nLandmarks );
		
      if( computeAffine )
      {
         aMatrix = new float[ndims][ndims];
         bVector = new float[ndims];
      }

      // dont reinitialize displacements if they already exist
      // for they may store residual errors after affine part.
      if(displacement == null){
         displacement = new float[nLandmarks][ndims];
      }
	}

   private void initMatricesAffine()
	{
		
		lMatrix = new DenseMatrix64F( ( (nLandmarks) * ndims ), ndims * ( ndims + 1 ) );
		yMatrix = new DenseMatrix64F( nLandmarks * ndims, 1 );
		wMatrix = new DenseMatrix64F( ndims * ( ndims + 1 ), 1 );
		
		dMatrix = new DenseMatrix64F( ndims, nLandmarks );
		
      if( computeAffine )
      {
         aMatrix = new float[ndims][ndims];
         bVector = new float[ndims];
      }
		displacement = new float[nLandmarks][ndims];
	}


   protected void computeAffine()
   {
      initMatricesAffine();
      computeD();
      computeAffineL();   
      computeY();

      logger.debug("lMatrix:\n " + lMatrix + "\n");
      logger.debug("yMatrix:\n " + yMatrix + "\n");

      LinearSolver<DenseMatrix64F> solver = 
         //LinearSolverFactory.general(lMatrix.numRows, lMatrix.numCols);
         LinearSolverFactory.leastSquares(lMatrix.numRows, lMatrix.numCols);

      solver.setA(lMatrix);
      solver.solve(yMatrix, wMatrix);
      logger.debug("wMatrix: " + wMatrix + "\n");

      reorganizeWAffine();

   }

   protected void computeAffineL(){
      for( int d=0; d<ndims; d++ ) 
      {
         for( int i=0; i<nLandmarks; i++ )
         {
            for( int j=0; j<ndims; j++ )
            {
               lMatrix.set( (i*ndims)+j, ndims*d +j, sourceLandmarks[d][i] );
               lMatrix.set( ( i*ndims) + j, ndims*ndims + j, 1 );
            }
         }
      }
   }

   protected void updateDisplacementPostAffine()
   {

      float[] srcPtTmp = new float[ndims];
      for( int i=0; i<nLandmarks; i++ ){

         for( int d=0; d<ndims; d++ ) {
            srcPtTmp[d] = sourceLandmarks[d][i];
         }
         float[] srcPtXfm = transformPointAffine(srcPtTmp); 

         for( int d=0; d<ndims; d++ ) {
            displacement[i][d] = targetLandmarks[d][i] - srcPtXfm[d];
         }
      }
   }

	protected void reorganizeWAffine(){
      int ci = 0;
      for( int dim=0; dim<ndims; dim++) 
      {
         // the affine part of the transform
         for( int j=0; j<ndims; j++) {
            aMatrix[dim][j] =  (float)wMatrix.get(ci,0);
            ci++;
         }
      }

      for( int dim=0; dim<ndims; dim++) 
      {
         // the translation part of the transform
         bVector[dim] = (float)wMatrix.get(ci, 0);
         ci++;

      }

      logger.debug(" affine:\n" + printArray(aMatrix));
      logger.debug(" b:\n" + printArray(bVector) +"\n");


		wMatrix = null;
		yMatrix = null;
		lMatrix = null;
	}
   
	protected float[] computeDeformationContribution( float[] thispt ){

		float[] result = new float[ndims];
		computeDeformationContribution( thispt, result ); 
		return result;
	}

	public float[] computeDeformationContribution( float[] thispt, float[] result ){

		// TODO: check for bugs - is l1 ever used?
		//double[] l1 = null;
		
		logger.debug("dMatrix: " + dMatrix);

		for( int lnd=0; lnd<nLandmarks; lnd++){
			
			double val = computeG( result );
			for (int i=0; i<ndims; i++) {
				result[i] += val * dMatrix.get(i,lnd);
			}
		}
		return result;
	}

	protected void computeD(){
		for( int d=0; d<ndims; d++ ) for( int i=0; i<nLandmarks; i++ ){
			displacement[i][d] = targetLandmarks[d][i] - sourceLandmarks[d][i]; 
		}
	}	

	protected float normSqrd( float[] v ){
     float nrm = 0;
      for(int i=0; i<v.length; i++){
         nrm += v[i]*v[i]; 
      }
      return nrm;
   }

	protected void computePostAffineDef(){

      // need to do this so that the 
      // reorganizeW call works properly
      computeAffine = false;

      computeWDef();

      // since we have an affine, though, we want this 
      // set to (true) in the end
      computeAffine = true;
   }

	/**
	 * The main workhorse method.
	 * <p>
	 * Implements Equation (5) in Davis et al.
	 * and calls reorganizeW.
	 *
	 */
	protected void computeWDef(){
		
		initMatricesDef();
	
      logger.debug("lMatrix:\n " + lMatrix + "\n");
      logger.debug("yMatrix:\n " + yMatrix + "\n");
      logger.debug("wMatrix:\n " + wMatrix + "\n");

      // assume this has already happened
		//computeD(); // only compute D once

		for (int d =0; d<ndims; d++)
		{
			// clear any previous data in the matrices
			yMatrix.zero();
			lMatrix.zero();
			wMatrix.zero();
			
			computeK(d);
			computeY(d);

			logger.debug(" lMatrix: " + lMatrix);
			logger.debug(" yMatrix: " + yMatrix);

			// solve linear system 
			LinearSolver<DenseMatrix64F> solver = null;

			// use pseudoinverse for underdetermined system
			// linear solver otherwise
			if( nLandmarks < ndims*ndims )
			{
				logger.debug("pseudo - inverse solver");
				solver =  LinearSolverFactory.pseudoInverse(true);
			}else
			{
				logger.debug("linear solver");
				solver =  LinearSolverFactory.linear(lMatrix.numCols);
			}

			// the general solver appears to work about as well as the linear solver
			//		LinearSolverFactory.general(lMatrix.numRows, lMatrix.numCols);

			solver.setA(lMatrix);
			solver.solve(yMatrix, wMatrix);

			logger.debug("wMatrix:\n" + wMatrix );

			reorganizeW(d);
		}
	}


	/**
	 * Builds the K matrix from landmark points and G matrix
	 * but drops the results directly into the L matrix.
	 */
	protected void computeK(int dim){

		float[] res = new float[ndims];

		for( int i=0; i<nLandmarks; i++ ){

			lMatrix.set(i, i, stiffness);

			for( int j = i+1; j<nLandmarks; j++ ){

				srcPtDisplacement(i,j,res);
				double val = computeG(res);

				lMatrix.set( i, j, val );
				lMatrix.set( j, i, val );
			}
		}
		logger.debug(" kMatrix: \n" + lMatrix + "\n");
	}

	protected DenseMatrix64F computeReflexiveG(){
		CommonOps.fill(gMatrix, 0);
		for (int i=0; i<ndims; i++){
			gMatrix.set(i,i, stiffness);
		}
		return gMatrix;
	}

	protected void computeY(){
      int k = 0;
		for (int i=0; i<nLandmarks; i++) 
		{
		   for (int d=0; d<ndims; d++) 
         {
			   yMatrix.set( k, 0, displacement[i][d]);
            k++;
         }
		}
   }

	/**
	 * Fills the y matrix with the landmark point displacements.
	 */
	protected void computeY(int dim){
		for (int i=0; i<nLandmarks; i++) 
		{
			yMatrix.set( i, 0, displacement[i][dim]);
		}
	}

	/**
	 * Copies data from the W matrix to the D, A, and b matrices
	 * which represent the deformable, affine and translational
	 * portions of the transformation, respectively.
	 */
	protected void reorganizeW(int dim){
		// TODO : this
		
		int ci = 0;

		// the deformable (non-affine) part of the transform
		for( int lnd=0; lnd<nLandmarks; lnd++){
			dMatrix.set(dim, lnd, wMatrix.get(ci, 0));
			ci++;
		}
		logger.debug(" dMatrix:\n" + dMatrix);

      if( computeAffine ) 
      {
         // the affine part of the transform
         for( int j=0; j<ndims; j++) {
            aMatrix[dim][j] =  (float)wMatrix.get(ci,0);
            ci++;
         }
         logger.debug(" affine:\n" + printArray(aMatrix));

         // the translation part of the transform
         
         bVector[dim] = (float)wMatrix.get(ci, 0);
         ci++;

		   logger.debug(" b:\n" + printArray(bVector) +"\n");
      }

      	logger.debug(" ");
      	
		//wMatrix = null;
		//yMatrix = null;
		//lMatrix = null;
	}

	public String printArray(float[] a){
		String out = "";
		for(int i=0; i<a.length; i++){
			out += a[i] + " ";
		}	
		out += "\n";
		return out;
	}
	public String printArray(float[][] a){
		String out = "";
		for(int i=0; i<a.length; i++){
			for(int j=0; j<a[0].length; j++){
				out += a[i][j] + " ";
			}	
			out += "\n";
		}
		return out;
	}

   /**
    * Transforms the input point according to the affine part of 
    * the thin plate spline stored by this object.  
    *
    * @param pt the point to be transformed
    * @return the transformed point
    */
	public float[] transformPointAffine(float[] pt){

      float[] result = new float[ndims];
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
	 * Transforms the input point according to the
	 * thin plate spline stored by this object.  
	 *
	 * @param pt the point to be transformed
	 * @return the transformed point
	 */
   public float[] transformPoint(float[] pt){
		
	  logger.trace("transforming pt:  " + printArray(pt));
	  float[] result = computeDeformationContribution( pt );
	  
	  logger.trace("res after def:   " + printArray(result));
	  
      if(aMatrix != null){
         // affine part
         for (int i = 0; i < ndims; i++) for (int j = 0; j < ndims; j++) {
            result[i] += aMatrix[i][j] * pt[j];
         }
      }
      logger.trace("res after aff:   " + printArray(result));
      
      if(bVector != null){
         // translational part
         for(int i=0; i<ndims; i++){
            result[i] += bVector[i] + pt[i];
         }
      }else{
    	  for (int i = 0; i < ndims; i++) 
    	  {
    		  result[i] += pt[i];
    	  }
      }
      logger.trace("res after trn:   " + printArray(result));

		return result;
	}

   /**
    *
    */
   public void transformInPlace(float[] pt)
   {
      float[] ptCopy = new float[ndims];
      for (int i = 0; i < ndims; i++){
         ptCopy[i] = pt[i];
      }

      float[] result = computeDeformationContribution( pt );
      for (int i = 0; i < ndims; i++){
         pt[i] = result[i];
      }

      if(aMatrix != null){
         // affine part
         for (int i = 0; i < ndims; i++) for (int j = 0; j < ndims; j++) {
            pt[i] += aMatrix[i][j] * ptCopy[j];
         }
      }else{
    	  for (int i = 0; i < ndims; i++) 
    	  {
    		  pt[i] += ptCopy[i];
    	  }
      }

      if(bVector != null){
         // translational part
         for(int i=0; i<ndims; i++){
            pt[i] += bVector[i] + ptCopy[i];
         }
      }

   }
	
	/**
	 * Computes the displacement between the i^th and j^th source points.
	 *
	 * Stores the result in the input array 'res'
	 * Does not validate inputs.
	 */
	protected void srcPtDisplacement(int i, int j, float[] res)
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
	protected void srcPtDisplacement(int i, float[] pt, float[] res)
	{
		for( int d=0; d<ndims; d++ ){
			res[d] = sourceLandmarks[d][i] - pt[d];
		}
	}
	protected float[] subtract(float[] p1, float[] p2){
		int nd = p1.length; 
		float[] out = new float[nd];
		for (int d=0; d<nd; d++){
			out[d] = p1[d] - p2[d];
		}
		return out;
	}
	
	protected float[] subtract(float[] p1, float[] p2, float[] out){
		int nd = out.length; 
		for (int d=0; d<nd; d++){
			out[d] = p1[d] - p2[d];
		}
		return out;
	}

	public abstract double computeG( float[] pt );


}

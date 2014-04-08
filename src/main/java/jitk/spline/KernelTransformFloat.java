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
 * “A physics-based coordinate transformation for 3-D image matching.,” 
 * IEEE Trans. Med. Imaging, vol. 16, no. 3, pp. 317–28, Jun. 1997. 
 *
 * @author Kitware (ITK)
 * @author John Bogovic
 *
 */
public abstract class KernelTransformFloat {
	
   protected int ndims;

	protected DenseMatrix64F gMatrix;
	protected DenseMatrix64F pMatrix;
	protected DenseMatrix64F kMatrix;
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
	
	protected static Logger logger = LogManager.getLogger(KernelTransformFloat.class.getName());
	
	//TODO: Many of these methods could be optimized by performing them without
	// explicit construction / multiplication of the matrices. 
	public KernelTransformFloat(){}

   /*
    * Constructor
    */
	public KernelTransformFloat(int ndims){
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
	public KernelTransformFloat( int ndims, float[][] srcPts, float[][] tgtPts){
		this(ndims);
		setLandmarks(srcPts, tgtPts);
	}

   /**
    * Constructor with transformation parameters.
    * aMatrix and bVector are allowed to be null
    */
   public KernelTransformFloat( float[][] srcPts, float[][] aMatrix, float[] bVector, double[] dMatrixData )
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
   
   private void initMatrices()
	{
		pMatrix = new DenseMatrix64F( (ndims * nLandmarks), ( ndims * (ndims + 1)) );
		dMatrix = new DenseMatrix64F( ndims, nLandmarks);
		wMatrix = new DenseMatrix64F( ndims, nLandmarks);
		kMatrix = new DenseMatrix64F( ndims * nLandmarks, ndims * nLandmarks);
		yMatrix = new DenseMatrix64F( ndims * ( nLandmarks + ndims + 1), 1 );
		lMatrix = new DenseMatrix64F( ndims * ( nLandmarks + ndims + 1),
									  ndims * ( nLandmarks + ndims + 1) );
		wMatrix = new DenseMatrix64F( (ndims * nLandmarks) + ndims * ( ndims + 1),
				  					  1 );
		
      if( computeAffine )
      {
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
			
			computeG( result, gMatrix );
			
			for (int i=0; i<ndims; i++) for (int j=0; j<ndims; j++){
				result[j] += gMatrix.get(i,j) * dMatrix.get(i,lnd);
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

      logger.debug(" lMatrix: " + lMatrix);
      logger.debug(" yMatrix: " + yMatrix);

		// solve linear system 
		LinearSolver<DenseMatrix64F> solver =  LinearSolverFactory.pseudoInverse(true);
		solver.setA(lMatrix);
		solver.solve(yMatrix, wMatrix);

		logger.debug("wMatrix:\n" + wMatrix );
		
		reorganizeW();
		
	}


	protected void computeL(){

      // fill P matrix if the affine parameters need to be computed
      if(computeAffine)
      {
         computeP();
      }
      // P matrix should be zero if points are already affinely aligned 
      
		computeK();

		CommonOps.insert( kMatrix, lMatrix, 0, 0 );
		CommonOps.insert( pMatrix, lMatrix, 0, kMatrix.getNumCols() );
		CommonOps.transpose(pMatrix);

		CommonOps.insert( pMatrix, lMatrix, kMatrix.getNumRows(), 0 );
		CommonOps.insert( kMatrix, lMatrix, 0, 0 );
	
      // bottom left O2 is already zeros after initializing 'lMatrix'
		
      
	}

	protected void computeP(){
		System.out.println("srcPts: " + sourceLandmarks);	
		DenseMatrix64F tmp = new DenseMatrix64F(ndims,ndims);

		for( int i=0; i<nLandmarks; i++ ){

			for( int d=0; d<ndims; d++ ){

				CommonOps.scale( sourceLandmarks[d][i], I, tmp);
				CommonOps.insert( tmp, pMatrix,  i*ndims, d*ndims );

			}
			CommonOps.insert( I, pMatrix,  i*ndims, ndims*ndims );
		}
	}


	/**
	 * Builds the K matrix from landmark points
	 * and G matrix.
	 */
	protected void computeK(){

		computeD();

		float[] res = new float[ndims];

		for( int i=0; i<nLandmarks; i++ ){

			DenseMatrix64F G = computeReflexiveG();
			CommonOps.insert(G, kMatrix, i * ndims, i * ndims);

			for( int j = i+1; j<nLandmarks; j++ ){

				srcPtDisplacement(i,j,res);
				computeG(res, G);

				CommonOps.insert(G, kMatrix, i * ndims, j * ndims);
				CommonOps.insert(G, kMatrix, j * ndims, i * ndims);
			}
		}
		logger.debug(" kMatrix: \n" + kMatrix + "\n");
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
		logger.debug(" dMatrix:\n" + dMatrix);

      if( computeAffine ) 
      {
         // the affine part of the transform
         for( int j=0; j<ndims; j++) for (int i=0; i<ndims; i++) {
            aMatrix[i][j] =  (float)wMatrix.get(ci,0);
            ci++;
         }
         logger.debug(" affine:\n" + printArray(aMatrix));

         // the translation part of the transform
         for( int k=0; k<ndims; k++) {
            bVector[k] = (float)wMatrix.get(ci, 0);
            ci++;
         }
		   logger.debug(" b:\n" + printArray(bVector) +"\n");
      }

		wMatrix = null;
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
	public double[] transformPointAffine(double[] pt){

      double[] result = new double[ndims];
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

	public abstract void computeG( float[] pt, DenseMatrix64F mtx);


}

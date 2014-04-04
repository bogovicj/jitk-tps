package jitk.spline;

import org.junit.Test;

import static org.junit.Assert.*;


public class ThinPlateR2LogRSplineKernelTransformTest {
	
	public static double tol = 0.0001;


	public static double[][] genPtListScale( double[][] srcPts , double[] scales ){
		double[][] pts = new double[srcPts.length][srcPts[0].length];
		
		for(int d=0; d<srcPts.length; d++) for (int n=0; n<srcPts[0].length; n++) {
			pts[d][n] = scales[d] * srcPts[d][n]; 
		}

		return pts;
	}

	@Test 
	public void testIdentity(){
		int ndims = 3;
		
		double[][] pts = new double[][]
				{
				{-1,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{-1,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{-1,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};
		int nL = pts[0].length;
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tps.computeW();
		
		double[] testPt = new double[ndims];
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			double[] outPt = tps.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}
		
		ThinPlateR2LogRSplineKernelTransform tpsNA = new ThinPlateR2LogRSplineKernelTransform( ndims, pts, pts );
		tpsNA.setDoAffine(false);
		tpsNA.computeW();
		
		for( int n=0; n<nL; n++) {
			for( int i=0; i<ndims; i++) {
				testPt[i] = pts[i][n];
			}
			
			double[] outPt = tpsNA.transformPoint(testPt);
			for( int i=0; i<ndims; i++) {
				assertEquals("Identity transformation", pts[i][n], outPt[i], tol);
			}
		}
		
	}
	
	@Test
	public void testScale3d() {

		int ndims = 3;
		double[][] src_simple = new double[][]
				{
				{0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2}, // x
				{0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2}, // y
				{0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2}, // z
				};

		double[][] tgtPtList = genPtListScale(src_simple, new double[]{2, 0.5, 4});

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgtPtList );
		tps.computeW();


		double[] srcPt = new double[]{ 0.0f, 0.0f, 0.0f};
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);
		assertEquals("scale z1", 0, ptXfm[2], tol);

		srcPt = new double[]{ 0.5f, 0.5f, 0.5f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);
		assertEquals("scale z2", 2.00, ptXfm[2], tol);

		srcPt = new double[]{ 1.0f, 1.0f, 1.0f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);
		assertEquals("scale z3", 4.0, ptXfm[2], tol);
	}

	@Test
	public void testAffineDone() {
		
		int ndims = 2;
		double[][] src = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		
		double[][] tgt = new double[src.length][src[0].length];
		
		for(int i=0; i<src.length; i++)for(int j=0; j<src[0].length; j++){
			tgt[i][j] = src[i][j] + Math.random()*0.1;
		}
		
		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src, tgt );
		tps.setDoAffine(false);
		tps.computeW();
		
		//System.out.println(" aMatrix: (Expect all zeros)\n" + 
		//		printArray(tps.aMatrix) + "\n");
		assertTrue("aMatrix should be null", tps.aMatrix==null);
		
		//System.out.println(" bVector: (Expect all zeros)\n" + 
		//		printArray(tps.bVector) + "\n");
		assertTrue("bVector should be null", tps.bVector==null);


		//System.out.println(" dMatrix: (Expect non-zero)\n" + tps.dMatrix + "\n");
		boolean isNonZeroMtxElem = false;
      for(int i=0; i<tps.dMatrix.getNumElements(); i++)
      {
         isNonZeroMtxElem = isNonZeroMtxElem || ( tps.dMatrix.get(i) != 0 );
      }
      assertTrue(" dMatrix as non-zero element", isNonZeroMtxElem );
		
	}
	
	public static final String printArray(double[][] in){
		if(in==null) return "null";
		String out = "";
		for(int i=0; i<in.length; i++){
			for(int j=0; j<in[0].length; j++){
				out += in[i][j] +" ";
			}
			out +="\n";
		}
		return out;
	}
	
	public static final String printArray(double[] in){
		if(in==null) return "null";
		String out = "";
		for(int i=0; i<in.length; i++){
			out += in[i] +" ";
		}
		return out;
	}
	
	
	@Test
	public void testScale() {

		int ndims = 2;
		double[][] src_simple = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};

		double[][] tgtPtList = genPtListScale(src_simple, new double[]{2, 0.5});

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgtPtList );
		tps.computeW();


		double[] srcPt = new double[]{ 0.0f, 0.0f };
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x1", 0, ptXfm[0], tol);
		assertEquals("scale y1", 0, ptXfm[1], tol);

		srcPt = new double[]{ 0.5f, 0.5f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x2", 1.00, ptXfm[0], tol);
		assertEquals("scale y2", 0.25, ptXfm[1], tol);

		srcPt = new double[]{ 1.0f, 1.0f };
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("scale x3", 2.0, ptXfm[0], tol);
		assertEquals("scale y3", 0.5, ptXfm[1], tol);
	}

	@Test
	public void testWarp() {

		int ndims = 2;
		double[][] src_simple = new double[][]
				{
				{0,0,0,1,1,1,2,2,2}, // x
				{0,1,2,0,1,2,0,1,2}, // y
				};
		// target points
		double[][] tgt= new double[][]
				{
				{ -0.5, -0.5, -0.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0}, // x
				{ -0.5, 1.5, 2.0, -0.5, 1.5, 2.0, -0.5, 1.5, 2.0 } // y
				};

		ThinPlateR2LogRSplineKernelTransform tps = new ThinPlateR2LogRSplineKernelTransform( ndims, src_simple, tgt);
		tps.computeW();
		//tps.printLandmarks();

		double[] srcPt = new double[]{0.0f,0.0f};
		double[] ptXfm = tps.transformPoint(srcPt);
		assertEquals("warp x1", -0.5, ptXfm[0], tol);
		assertEquals("warp y1", -0.5, ptXfm[1], tol);

		srcPt = new double[]{0.5f,0.5f};
		ptXfm = tps.transformPoint(srcPt);
		
		// the values below are what matlab returns for
		// tpaps( p, q, 1 );
		// where p and q are the source and target points, respectively
		assertEquals("warp x2", 0.6241617, ptXfm[0], tol);
		assertEquals("warp y2", 0.6241617, ptXfm[1], tol);

		srcPt = new double[]{1.0f,1.0f};
		ptXfm = tps.transformPoint(srcPt);
		assertEquals("warp x3", 1.5, ptXfm[0], tol);
		assertEquals("warp y3", 1.5, ptXfm[1], tol);

	}

}

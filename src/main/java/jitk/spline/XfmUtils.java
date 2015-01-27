package jitk.spline;

/**
 * Class with helper methods for debugging and testing the
 * this thin plate spline implementation.
 * 
 * @author John Bogovic
 *
 */
public class XfmUtils {

	public static double[][] genPtListScale( double[][] srcPts , double[] scales ){
		double[][] pts = new double[srcPts.length][srcPts[0].length];
		
		for(int d=0; d<srcPts.length; d++) for (int n=0; n<srcPts[0].length; n++) {
			pts[d][n] = scales[d] * srcPts[d][n]; 
		}

		return pts;
	}
	public static float[][] genPtListScale( float[][] srcPts , double[] scales ){
		float[][] pts = new float[srcPts.length][srcPts[0].length];
		
		for(int d=0; d<srcPts.length; d++) for (int n=0; n<srcPts[0].length; n++) {
			pts[d][n] = (float)(scales[d] * srcPts[d][n]); 
		}

		return pts;
	}
	public static double[][] genPtListAffine( double[][] srcPts , double[][] aff ){
		double[][] pts = new double[srcPts.length][srcPts[0].length];
		
		for (int n=0; n<srcPts[0].length; n++) {
			for(int i=0; i<srcPts.length; i++) for(int j=0; j<srcPts.length; j++)
			{
				pts[i][n] = (aff[i][j] * srcPts[j][n]);
			}
		}

		return pts;
	}
	public static float[][] genPtListAffine( float[][] srcPts , float[][] aff ){
		float[][] pts = new float[srcPts.length][srcPts[0].length];
		
		for (int n=0; n<srcPts[0].length; n++) {
			for(int i=0; i<srcPts.length; i++) for(int j=0; j<srcPts.length; j++)
			{
				pts[i][n] = (aff[i][j] * srcPts[j][n]);
			}
		}

		return pts;
	}
	
	public static double[] subtract(double[] p1, double[] p2){
		int nd = p1.length; 
		double[] out = new double[nd];
		for (int d=0; d<nd; d++){
			out[d] = p1[d] - p2[d];
		}
		return out;
	}
	public static float[] subtract(float[] p1, float[] p2){
		int nd = p1.length; 
		float[] out = new float[nd];
		for (int d=0; d<nd; d++){
			out[d] = p1[d] - p2[d];
		}
		return out;
	}
	
	public static double[] subtract(double[] p1, double[] p2, double[] out){
		int nd = out.length; 
		for (int d=0; d<nd; d++){
			out[d] = p1[d] - p2[d];
		}
		return out;
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
	
	public static final String printArray(float[][] in){
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
	
	public static final String printArray(float[] in){
		if(in==null) return "null";
		String out = "";
		for(int i=0; i<in.length; i++){
			out += in[i] +" ";
		}
		return out;
	}
}

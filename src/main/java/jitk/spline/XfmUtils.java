package jitk.spline;

import mpicbg.models.CoordinateTransform;

/**
 * Class with helper methods for debugging and testing the this thin plate
 * spline implementation.
 * 
 * @author John Bogovic
 * 
 */
public class XfmUtils {

	public static double[][] genPtListScale(double[][] srcPts, double[] scales) {
		double[][] pts = new double[srcPts.length][srcPts[0].length];

		for (int d = 0; d < srcPts.length; d++)
			for (int n = 0; n < srcPts[0].length; n++) {
				pts[d][n] = scales[d] * srcPts[d][n];
			}

		return pts;
	}

	public static float[][] genPtListScale(float[][] srcPts, double[] scales) {
		float[][] pts = new float[srcPts.length][srcPts[0].length];

		for (int d = 0; d < srcPts.length; d++)
			for (int n = 0; n < srcPts[0].length; n++) {
				pts[d][n] = (float) (scales[d] * srcPts[d][n]);
			}

		return pts;
	}

	public static double[][] genPtListAffine(double[][] srcPts, double[][] aff) {
		double[][] pts = new double[srcPts.length][srcPts[0].length];

		for (int n = 0; n < srcPts[0].length; n++) {
			for (int i = 0; i < srcPts.length; i++)
				for (int j = 0; j < srcPts.length; j++) {
					pts[i][n] = (aff[i][j] * srcPts[j][n]);
				}
		}

		return pts;
	}

	public static float[][] genPtListAffine(float[][] srcPts, float[][] aff) {
		float[][] pts = new float[srcPts.length][srcPts[0].length];

		for (int n = 0; n < srcPts[0].length; n++) {
			for (int i = 0; i < srcPts.length; i++)
				for (int j = 0; j < srcPts.length; j++) {
					pts[i][n] = (aff[i][j] * srcPts[j][n]);
				}
		}

		return pts;
	}
	
	public static double maxError( double[][] sourcePoints, double[][] targetPoints, CoordinateTransform xfm )
	{
		return maxError( sourcePoints, targetPoints, xfm, false );
	}
	
	public static double maxError( double[][] sourcePoints, double[][] targetPoints, CoordinateTransform xfm, boolean debug )
	{
		double maxError = 0.0;
		
		int D = sourcePoints.length;
		int N = sourcePoints[ 0 ].length;
		
		double[] spt = new double[ sourcePoints.length ];
		double[] tpt = new double[ sourcePoints.length ];
		for( int i = 0; i < N; i++ )
		{
			for( int n = 0; n < D; n++ )
			{
				spt[ n ] = sourcePoints[ n ][ i ];
				tpt[ n ] = targetPoints[ n ][ i ];
			}
				
			double[] result = xfm.apply( spt );
			double distance = distance( tpt, result );
			
			if( debug )
				System.out.println( "distance at: " + i + " is " + distance );
			
			if( distance > maxError )
				maxError = distance;
		}
		
		return maxError;
	}
	
	public static double[][] deepCopy( double[][] in ) {
		double[][] out = new double[ in.length ][ in[0].length ];
		for (int i = 0; i < in.length; i++)for (int j = 0; j < in[0].length; j++) {
			out[i][j] = in[i][j]; 
		}
		return out;
	}

	public static double[] toDouble(float[] in) {
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++) {
			out[i] = in[i];
		}
		return out;
	}

	public static float[] toFloat(double[] in) {
		float[] out = new float[in.length];
		for (int i = 0; i < in.length; i++) {
			out[i] = (float) in[i];
		}
		return out;
	}

	/**
	 * Copy from src to dest
	 * 
	 * @param src
	 * @param dest
	 */
	public static void copy(float[] src, double[] dest) {
		for (int i = 0; i < src.length; i++) {
			dest[i] = src[i];
		}
	}

	/**
	 * Copy from src to dest
	 * 
	 * @param src
	 * @param dest
	 */
	public static void copy(double[] src, float[] dest) {
		for (int i = 0; i < src.length; i++) {
			dest[i] = (float) src[i];
		}
	}

	public static double distance( double[] p1, double[] p2 )
	{
		int nd = p1.length;
		double out = 0.0;
		for (int d = 0; d < nd; d++) {
			out += ( p1[d] - p2[d] ) * ( p1[d] - p2[d] );
		}
		
		return Math.sqrt( out );
	}
	
	public static double[] subtract(double[] p1, double[] p2) {
		int nd = p1.length;
		double[] out = new double[nd];
		for (int d = 0; d < nd; d++) {
			out[d] = p1[d] - p2[d];
		}
		return out;
	}

	public static double[] subtract(double[] p1, double[] p2, double[] out) {
		int nd = out.length;
		for (int d = 0; d < nd; d++) {
			out[d] = p1[d] - p2[d];
		}
		return out;
	}

	public static final String printArray(double[][] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[0].length; j++) {
				out += in[i][j] + " ";
			}
			out += "\n";
		}
		return out;
	}

	public static final String printArray(double[] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			out += in[i] + " ";
		}
		return out;
	}

	public static final String printArray(boolean[] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			if( in[i])
				out += "1 ";
			else
				out += "0 ";
		}
		return out;
	}
	
	public static final String printArray(float[][] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[0].length; j++) {
				out += in[i][j] + " ";
			}
			out += "\n";
		}
		return out;
	}

	public static final String printArray(float[] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			out += in[i] + " ";
		}
		return out;
	}

	public static final String printArray(long[][] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[0].length; j++) {
				out += in[i][j] + " ";
			}
			out += "\n";
		}
		return out;
	}

	public static final String printArray(long[] in) {
		if (in == null)
			return "null";
		String out = "";
		for (int i = 0; i < in.length; i++) {
			out += in[i] + " ";
		}
		return out;
	}
}

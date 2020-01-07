package hex.gam.MatrixFrameUtils;

import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.GamSplines.CubicRegressionSplines;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;

public class GenerateGamMatrixOneColumn extends MRTask<GenerateGamMatrixOneColumn> {
  BSType _splineType;
  int _numKnots;      // number of knots
  double[][] _bInvD;  // store inv(B)*D
  double _mean;
  double _oneOSigma;
  Frame _gamX;
  
  public GenerateGamMatrixOneColumn(BSType splineType, int numKnots, double[] knots, Frame gamx, boolean standardize) {
    _splineType = splineType;
    _numKnots = numKnots;
    _mean = standardize?gamx.vec(0).mean():0;
    _oneOSigma = 1.0/(standardize?gamx.vec(0).sigma():1);
    CubicRegressionSplines crSplines = new CubicRegressionSplines(numKnots, knots, gamx.vec(0).max(), gamx.vec(0).min());
    _bInvD = crSplines.genreateBIndvD(crSplines._hj);
    _gamX = gamx;
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newGamCols) {
    int chunkRows = chk[0].len(); // number of rows in chunk
    CubicRegressionSplines crSplines = new CubicRegressionSplines(_numKnots, null, _gamX.vec(0).max(), 
            _gamX.vec(0).min());
    double[] basisVals = new double[_numKnots];
    for (int rowIndex=0; rowIndex < chunkRows; rowIndex++) {
     // find index of knot bin where row value belongs to
     double xval = chk[0].atd(rowIndex);
     int binIndex = locateBin(xval,crSplines._knots); // location to update
      if (binIndex == 5)
        System.out.println("Wow");
      // update from F matrix F matrix = [0;invB*D;0] and c functions
      updateFMatrixCFunc(basisVals, xval, binIndex, crSplines, _bInvD);
      // update from a functions
      updateAFunc(basisVals, xval, binIndex, crSplines);
      // copy updates to the newChunk row
      for (int colIndex = 0; colIndex < _numKnots; colIndex++)
        newGamCols[colIndex].addNum(basisVals[colIndex]);
    }
  }
  
  public static void updateAFunc(double[] basisVals, double xval, int binIndex, CubicRegressionSplines splines) {
    int jp1 = binIndex+1;
    basisVals[binIndex] += splines.gen_a_m_j(splines._knots[jp1], xval, splines._hj[binIndex]);
    basisVals[jp1] += splines.gen_a_p_j(splines._knots[binIndex], xval, splines._hj[binIndex]);
  }
  
  public static void updateFMatrixCFunc(double[] basisVals, double xval, int binIndex, CubicRegressionSplines splines,
                                        double[][] binvD) {
    int numKnots = basisVals.length;
    int matSize = binvD.length;
    int jp1 = binIndex+1;
    double cmj = splines.gen_c_m_j(splines._knots[jp1], xval, splines._hj[binIndex]);
    double cpj = splines.gen_c_p_j(splines._knots[binIndex], xval, splines._hj[binIndex]);
    int binIndexM1 = binIndex-1;
    for (int index=0; index < numKnots; index++) {
      if (binIndex == 0) {  // only one part
        basisVals[index] = binvD[binIndex][index] * cpj;
      } else if (binIndex >= matSize) { // update only one part
        basisVals[index] = binvD[binIndexM1][index] * cmj;
      } else { // binIndex > 0 and binIndex < matSize
        basisVals[index] = binvD[binIndexM1][index] * cmj;
        basisVals[index] += binvD[binIndex][index] * cpj;
      }
    }
  }
  
  public static int locateBin(double xval, double[] knots) {
    if (xval <= knots[0])  //small short cut
      return 0;
    int highIndex = knots.length-1;
    if (xval >= knots[highIndex]) // small short cut
      return (highIndex-1);
    
    int binIndex = 0; 
    int count = 0;
    int numBins = knots.length;
    int lowIndex = 0;
    
    while (count < numBins) {
      int tryBin = (int) Math.floor((highIndex+lowIndex)*0.5);
      if ((xval >= knots[tryBin]) && (xval < knots[tryBin+1]))
        return tryBin;
      else if (xval > knots[tryBin])
        lowIndex = tryBin;
      else if (xval < knots[tryBin])
        highIndex = tryBin;
      
      count++;
    }
    return binIndex;
  }
}



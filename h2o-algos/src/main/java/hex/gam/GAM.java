package hex.gam;

import hex.ModelBuilder;
import hex.ModelCategory;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.Link;
import jsr166y.RecursiveAction;
import jsr166y.ForkJoinTask;
import water.Key;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import hex.gam.GAMModel.GAMParameters.BSType;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import water.DKV;
import water.Scope;

import java.util.Arrays;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() { return new ModelCategory[]{ModelCategory.Regression}; }

  @Override
  public boolean isSupervised() { return true; }

  @Override
  public BuilderVisibility builderVisibility() { return BuilderVisibility.Experimental; }
  
  @Override public boolean havePojo() { return false; }
  @Override public boolean haveMojo() { return false; }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }
  public GAM(GAMModel.GAMParameters parms) { super(parms);init(false); }
  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) { super(parms, key); init(false); }

  @Override public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (!_parms._family.equals(Family.gaussian)) 
        error("_family", "Only gaussian family is supported for now.");
      if (!_parms._link.equals(Link.identity) && !_parms._link.equals(Link.family_default))
        error("_link", "Only identity or family_default link is supported for now.");
      if (_parms._gam_X==null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k==null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10?(int)_train.numRows():10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      }
    }
  }

  @Override
  public void checkDistributions() {  // will be called in ModelBuilder.java
    if (!_response.isNumeric()) {
      error("_response", "Expected a numerical response, but instead got response with " + _response.cardinality() + " categories.");
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return false; // no use, we don't output probabilities
  }

  @Override
  protected int init_getNClass() {
    return 1; // only regression is supported for now
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }
  
  @Override 
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override protected void checkMemoryFootPrint_impl() {
    ;
  }
  private class GAMDriver extends Driver {
    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * @return
     */
    Frame adaptTrain(GAMModel model) {
      Frame orig = _parms.train();  // contain all columns, _train contains only predictors and responses
      Scope.track(orig);
      int numGamFrame = _parms._gam_X.length;
      Key<Frame>[] gamFramesKey = new Key[numGamFrame];  // store the Frame keys of generated GAM column
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      final boolean centerGam = (numGamFrame > 1) || (_train.numCols()-1+numGamFrame) >= 2; // only need to center when predictors > 1
      if (centerGam && _parms._saveZMatrix) {
        model._output._zTranspose = new double[numGamFrame][][];
        for (int frameIdx = 0; frameIdx < numGamFrame; frameIdx++) {
          int numKnots = _parms._k[frameIdx];
          model._output._zTranspose[frameIdx] = new double[numKnots-1][numKnots];
        }
      }
      for (int index=0; index < numGamFrame; index++) {
        final Frame predictVec = new Frame(orig.vec(_parms._gam_X[index]));  // extract the vector to work on
        final int numKnots = _parms._k[index];  // grab number of knots to generate
        final BSType splineType = _parms._bs[index];
        final int tIndex = index;
        final String[] newColNames = new String[numKnots];
        for (int colIndex = 0; colIndex < numKnots; colIndex++) {
          newColNames[colIndex] = _parms._gam_X[index]+"_"+splineType.toString()+"_"+colIndex;
        }
        generateGamColumn[tIndex] = new RecursiveAction() {
          @Override
          protected void compute() {
            GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots, null, predictVec,
                    _parms._standardize, centerGam).doAll(numKnots,Vec.T_NUM,predictVec);
            Frame oneAugmentedColumn = genOneGamCol.outputFrame(Key.make(), newColNames,
                    null);
/*            if (centerGam)
              oneAugmentedColumn = genOneGamCol.de_centralize_frame(oneAugmentedColumn, 
                      predictVec.name(0)+"_"+splineType.toString()+"_decenter_", _parms);*/
            Scope.track(oneAugmentedColumn); // track for automatic removal from DKV
            if (_parms._saveZMatrix) {
              int numZTCol = numKnots-1;
              for (int colIdx=0; colIdx < numZTCol; colIdx++) { // save zMatrix for debugging purposes or later scoring on training dataset
                System.arraycopy(genOneGamCol._ZTransp[colIdx], 0, model._output._zTranspose[tIndex][colIdx], 0,
                        genOneGamCol._ZTransp[colIdx].length);
              }
            }
            Scope.track(oneAugmentedColumn);  // track frame with new frame.key
            gamFramesKey[tIndex] = oneAugmentedColumn._key;
            DKV.put(oneAugmentedColumn);
          }
        };
      }
      ForkJoinTask.invokeAll(generateGamColumn);

      if (_parms._saveGamCols)  // save generated Gam Columns for debugging purposes
        model._output._gamX  = new Key[numGamFrame];
      for (int frameInd = 0; frameInd < numGamFrame; frameInd++) {  // append the augmented columns to _train
        _train.add(gamFramesKey[frameInd].get());
        if (_parms._saveGamCols)
          model._output._gamX[frameInd] = gamFramesKey[frameInd];
      }
      Scope.track(_train);
      return _train;
    }
    
    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      _job.update(0, "Initializing model training");
      
      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this));
      model.delete_and_lock(_job);

     Frame newTFrame = adaptTrain(model);  // get frames with correct predictors and spline functions
      Scope.track(newTFrame);

    }
  }
}

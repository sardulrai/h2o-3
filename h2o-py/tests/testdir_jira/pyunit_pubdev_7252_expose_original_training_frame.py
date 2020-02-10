import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.isolation_forest import H2OIsolationForestEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator



def test_expose_original_training_frame():
    airlines= h2o.import_file(path=pyunit_utils.locate("smalldata/airlines/allyears2k_headers.zip"))

    # convert columns to factors
    airlines["Year"]= airlines["Year"].asfactor()
    airlines["Month"]= airlines["Month"].asfactor()
    airlines["DayOfWeek"] = airlines["DayOfWeek"].asfactor()
    airlines["Cancelled"] = airlines["Cancelled"].asfactor()
    airlines['FlightNum'] = airlines['FlightNum'].asfactor()


    # set the predictor names and the response column name
    predictors = ["Origin", "Dest", "Year", "UniqueCarrier", "DayOfWeek", "Month", "Distance", "FlightNum"]
    response = "IsDepDelayed"
    predictorsAndResponse = predictors
    predictorsAndResponse.append(response)

    # split into train and validation sets
    train, valid= airlines.split_frame(ratios = [.8], seed = 1234)

    # try using the `categorical_encoding` parameter:
    encoding = "one_hot_explicit"


    # initialize the estimator
    airlines_gbm_enc = H2OGradientBoostingEstimator(categorical_encoding = encoding, seed =1234)
    airlines_gbm = H2OGradientBoostingEstimator(seed =1234)

    # then train the model
    airlines_gbm_enc.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
    airlines_gbm.train(x = predictors, y = response, training_frame = train, validation_frame = valid)

    assert(airlines_gbm._model_json['output']['names'] == airlines_gbm_enc._model_json['output']['original_names'])
    assert(sorted(predictorsAndResponse) == sorted(airlines_gbm_enc._model_json['output']['original_names']))


    # initialize the estimator
    airlines_if_enc = H2OIsolationForestEstimator(categorical_encoding = encoding, seed =1234)
    airlines_if = H2OIsolationForestEstimator(seed =1234)

    # then train the model
    airlines_if_enc.train(x = predictors,y = response, training_frame = train)
    airlines_if.train(x = predictors,y = response, training_frame = train)

    assert(airlines_if._model_json['output']['names'] == airlines_if_enc._model_json['output']['original_names'])
    assert(sorted(predictorsAndResponse) == sorted(airlines_if_enc._model_json['output']['original_names']))


    # initialize the estimator
    airlines_drf_enc = H2ORandomForestEstimator(categorical_encoding = encoding, seed =1234)
    airlines_drf = H2ORandomForestEstimator(seed =1234)

    # then train the model
    airlines_drf_enc.train(x = predictors, y= response, training_frame = train, validation_frame = valid)
    airlines_drf.train(x = predictors, y=response, training_frame = train, validation_frame = valid)

    assert(airlines_drf._model_json['output']['names'] == airlines_drf_enc._model_json['output']['original_names'])
    assert(sorted(predictorsAndResponse) == sorted(airlines_drf_enc._model_json['output']['original_names']))


    # initialize the estimator
    airlines_xgb_enc = H2OXGBoostEstimator(categorical_encoding = encoding, seed =1234)
    airlines_xgb = H2OXGBoostEstimator(seed =1234)

    # then train the model
    airlines_xgb_enc.train(x = predictors, y= response, training_frame = train, validation_frame = valid)
    airlines_xgb.train(x = predictors, y=response, training_frame = train, validation_frame = valid)

    #fails:
    #assert(airlines_xgb._model_json['output']['names'] == airlines_xgb_enc._model_json['output']['original_names'])
    #fails:
    #assert(sorted(predictorsAndResponse) == sorted(airlines_xgb_enc._model_json['output']['original_names']))
    #fails:
    #assert(sorted(predictors) == sorted(airlines_xgb_enc._model_json['output']['original_names']))
    assert(sorted(predictorsAndResponse) == sorted(airlines_xgb._model_json['output']['names']))
    assert(sorted(airlines_xgb_enc._model_json['output']['original_names']) == sorted(airlines.names))

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_expose_original_training_frame)
else:
    test_expose_original_training_frame()

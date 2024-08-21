from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import PipelineModel


def load_model():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("LoanDefaulterPrediction") \
        .getOrCreate()

    # Load trained model
    loaded_model = PipelineModel.load("./model")
    return spark, loaded_model

def predict_loan_default(spark, loaded_model, test_data):
    # Create a DataFrame from test data
    synthetic_test_data = spark.createDataFrame([test_data], 
        ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"])

    # Preprocess test data
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(synthetic_test_data) for column in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
    indexed_test_data = synthetic_test_data
    for indexer in indexers:
        indexed_test_data = indexer.transform(synthetic_test_data)

    # Create feature vector
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR_index', 'FLAG_OWN_REALTY_index']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    data_with_features_test = assembler.transform(indexed_test_data)

    # Make predictions on test data
    predictions_test = loaded_model.transform(data_with_features_test)
    result = predictions_test.select("prediction", "probability").collect()[0]
    
    # Stop SparkSession
    spark.stop()

    return {"prediction": result.prediction, "probability": result.probability}


from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StringIndexer

def test_model():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("LoanDefaulterPrediction") \
        .getOrCreate()

    # Load trained model
    loaded_model = CrossValidatorModel.load("./result")

    # Create synthetic test data
    synthetic_test_data = spark.createDataFrame([
        (202500.0, 406597.5, 24700.5, -637, -3648.0, "Y", "Y"),
        (270000.0, 1293502.5, 35698.5, -1188, -1186.0, "N", "N"),
        (90500.0, 135000.0, 6750.0, -225, -4260.0, "Y", "Y")
    ], ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"])

    # Preprocess test data (same preprocessing steps as training data)
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(synthetic_test_data) for column in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
    selected_test_data = synthetic_test_data.select(['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'])
    selected_test_data = selected_test_data.dropna()

    # Convert string columns to numerical categories
    indexed_test_data = selected_test_data
    for indexer in indexers:
        indexed_test_data = indexer.transform(indexed_test_data)

    # Create feature vector
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR_index', 'FLAG_OWN_REALTY_index']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    data_with_features_test = assembler.transform(indexed_test_data)

    # Make predictions on test data
    predictions_test = loaded_model.transform(data_with_features_test)

    # Display predictions
    predictions_test.select("prediction").show()

    # Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    test_model()

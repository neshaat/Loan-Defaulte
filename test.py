from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StringIndexer


def load_model():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("LoanDefaulterPrediction") \
        .getOrCreate()

    model = CrossValidatorModel.load("./model")

    return spark, model


def preprocess_data(spark):
    synthetic_test_data = spark.createDataFrame([
        (202500.0, 406597.5, 24700.5, -637, -3648.0, "Y", "Y"),
        (270000.0, 1293502.5, 35698.5, -1188, -1186.0, "N", "N"),
        (90500.0, 135000.0, 6750.0, -225, -4260.0, "Y", "Y")
    ], ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY"])

    indexers = [
        StringIndexer(inputCol="FLAG_OWN_CAR", outputCol="FLAG_OWN_CAR_index"),
        StringIndexer(inputCol="FLAG_OWN_REALTY", outputCol="FLAG_OWN_REALTY_index")
    ]

    for indexer in indexers:
        synthetic_test_data = indexer.fit(synthetic_test_data).transform(synthetic_test_data)

    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED',
                    'DAYS_REGISTRATION', 'FLAG_OWN_CAR_index', 'FLAG_OWN_REALTY_index']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    data_with_features_test = assembler.transform(synthetic_test_data)

    return data_with_features_test


def predict_loan_default(spark, model):
    data_with_features_test = preprocess_data(spark)
    predictions_test = model.transform(data_with_features_test)
    predictions_test.select("prediction").show()


if __name__ == "__main__":
    spark, model = load_model()

    predict_loan_default(spark, model)

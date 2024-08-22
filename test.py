from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col


def load_model():
    spark = SparkSession.builder \
        .appName("LoanDefaulterPrediction") \
        .getOrCreate()

    model_path = "./model"
    model = PipelineModel.load(model_path)
    return spark, model


def add_string_indexers(df, indexers):
    for input_col, output_col in indexers.items():
        if output_col not in df.columns:
            indexer = StringIndexer(inputCol=input_col, outputCol=output_col).fit(df)
            df = indexer.transform(df)
        else:
            print(f"Column {output_col} already exists. Skipping StringIndexer for {input_col}.")
    return df


def preprocess_data(spark, data):
    columns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR',
               'FLAG_OWN_REALTY']
    df = spark.createDataFrame([data], columns)
    indexers = {
        'FLAG_OWN_CAR': 'FLAG_OWN_CAR_index',
        'FLAG_OWN_REALTY': 'FLAG_OWN_REALTY_index'
    }
    df = add_string_indexers(df, indexers)
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                    'FLAG_OWN_CAR_index', 'FLAG_OWN_REALTY_index']

    for col_name in feature_cols:
        if col_name not in df.columns:
            raise ValueError(f"Missing feature column: {col_name}")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)
    return df


def predict_loan_default(spark, model, test_data):
    """
    Predicts loan default using the Spark model.
    """
    test_df = preprocess_data(spark, test_data)
    predictions = model.transform(test_df)
    result = predictions.select("prediction").collect()[0][0]
    return {"prediction": result}

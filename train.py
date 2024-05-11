from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

def train_model():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("LoanDefaulterPrediction") \
        .getOrCreate()

    # Load data
    data = spark.read.csv("./application_data.csv", header=True, inferSchema=True)

    # Select relevant columns and handle missing values
    selected_data = data.select(['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'])

    # Drop rows with missing values
    selected_data = selected_data.dropna()

    # Convert string columns to numerical categories
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(selected_data) for column in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
    indexed_data = selected_data
    for indexer in indexers:
        indexed_data = indexer.transform(indexed_data)

    # Create feature vector
    feature_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'FLAG_OWN_CAR_index', 'FLAG_OWN_REALTY_index']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    data_with_features = assembler.transform(indexed_data)

    # Split data into train and test sets
    train_data, test_data = data_with_features.randomSplit([0.7, 0.3], seed=42)

    # Build Random Forest model
    rf = RandomForestClassifier(labelCol='TARGET', featuresCol='features')

    # Define a parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 150]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()

    # Define evaluator
    evaluator = BinaryClassificationEvaluator(labelCol='TARGET')

    # Perform cross-validation
    cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    # Train model
    cv_model = cv.fit(train_data)

    # Save trained model
    cv_model.save("/content/result")

    # Stop SparkSession
    spark.stop()

if __name__ == "__main__":
    train_model()


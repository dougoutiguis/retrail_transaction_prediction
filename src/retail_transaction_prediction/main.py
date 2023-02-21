import utils.training_data_preparation_utils as tdpu
import features_creation as fc
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, MinMaxScaler, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from xgboost.spark import SparkXGBRegressor

if __name__ == '__main__':
    fcc = fc.FeaturesCreation()
    df = fcc.read_file("csv")
    # df.select(f.date_format("InvoiceDate", "yyyy-MM-dd").alias("Day")).distinct().orderBy("day").show(767, False)
    df_prepared = tdpu.prepare_data(df, old_name="Customer ID", new_name="CustomerId")
    #tdpu.get_buying_frequency(["2011-08-05", "2011-03-25", "2011-10-21", "2009-12-15", "2010-05-10", "2010-12-07", "2010-04-22", "2011-07-28", "2011-09-28", "2011-06-09", "2009-12-21", "2010-07-23", "2009-12-11", "2011-12-09", "2010-08-09", "2010-04-27", "2011-10-03", "2010-12-09", "2010-07-16", "2011-03-03", "2010-06-23", "2010-11-10", "2011-09-22", "2011-09-15", "2010-11-15", "2011-07-04", "2010-06-08", "2009-12-01", "2010-08-26", "2010-08-11", "2010-11-08", "2011-10-04", "2010-08-17", "2009-12-03", "2010-03-05", "2010-07-30", "2010-10-14", "2011-11-04", "2011-12-08", "2011-11-28", "2010-03-29", "2011-06-07", "2010-07-15", "2010-03-18", "2010-01-12", "2010-10-17", "2010-08-31", "2010-02-25", "2009-12-22", "2011-09-02", "2010-01-05", "2010-05-21", "2010-04-12", "2010-02-19", "2011-07-20", "2010-08-02", "2011-06-14", "2010-08-01", "2010-03-30", "2010-02-05", "2011-05-17", "2011-02-07", "2011-05-16", "2010-07-27", "2010-01-08", "2010-05-17", "2011-04-20"])
    purchaser_data = tdpu.get_purchaser_data(df_prepared, "2011-10")
    non_purchaser_data = tdpu.get_non_buyer_data(df_prepared, "2011-10")
    purchaser_data_11_2011 = tdpu.get_purchaser_data(df_prepared, "2011-11")
    non_purchaser_data_11_2011 = tdpu.get_non_buyer_data(df_prepared, "2011-11")
    purchasers_featured = fcc.add_features(purchaser_data)
    purchasers_featured_11_2011 = fcc.add_features(purchaser_data_11_2011)
    non_purchaser_featured = fcc.add_features(non_purchaser_data, False)
    non_purchasers_featured_11_2011 = fcc.add_features(non_purchaser_data_11_2011, False)
    data = fcc.get_data_prepared(df, ["2011-05", "2011-06", "2011-07", "2011-08", "2011-09"])
    # print(purchasers_featured.count(), non_purchaser_featured.count(), purchasers_featured.unionByName(non_purchaser_featured).count())
    # purchasers_featured.unionByName(non_purchaser_featured).show(30, False)

    # print(tdpu.get_purchaser_data(df_prepared, "2011-10").count(), tdpu.get_purchaser_data(df_prepared, "2011-10")
    #       .select("CustomerId").distinct().count(), tdpu.get_purchasers(df_prepared, "2011-10").count(),
    #       non_purchaser_data.select("CustomerId").distinct().count())

    # tdpu.get_purchaser_data(df_prepared, "2011-11").show(20, False)
    # print(tdpu.get_purchaser_data(df_prepared, "2011-11").count(), tdpu.get_purchaser_data(df_prepared, "2011-11")
    #       .select("CustomerId").distinct().count(), tdpu.get_purchasers(df_prepared, "2011-11").count())
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    featureScaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    input_cols = data.columns
    input_cols.remove("CustomerId")
    input_cols.remove("label")
    # input_cols = ['total_spend', 'total_distinct_items', 'total_buying_days', 'delay_between_2_buying_act', 'total_distinct_items_c']
    # print(input_cols)
    # input_cols
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features").transform(data)
    # assembler.show(5, False)
    featureIndexer = \
        VectorIndexer(inputCol="scaledFeatures", outputCol="indexedFeatures", maxCategories=4)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = assembler.randomSplit([0.7, 0.3])

    # Train a GBT model.
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

    # rft = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", maxIter=10)

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureScaler, featureIndexer, labelIndexer,  gbt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    assembler1 = VectorAssembler(inputCols=input_cols, outputCol="features")\
        .transform(purchasers_featured_11_2011.unionByName(non_purchasers_featured_11_2011))
    predictions_real = model.transform(assembler1)
    # Select example rows to display.
    # predictions_real.show(5)

    # Select (prediction, true label) and compute test error
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print("Test AUC = %g" % (auc))

    gbtModel = model.stages[1]
    print(gbtModel)  # summary only


    # spark_reg_estimator = SparkXGBRegressor(
    #     features_col="features",
    #     label_col="label",
    #     num_workers=2,
    # )
    # xgb_regressor_model = spark_reg_estimator.fit(trainingData)
    # transformed_test_spark_dataframe = spark_reg_estimator.predict(testData)
    # evaluator = MulticlassClassificationEvaluator(
    #     labelCol="label", predictionCol="prediction", metricName="accuracy")
    # accuracy = evaluator.evaluate(transformed_test_spark_dataframe)
    # print("Test Error = %g" % (1.0 - accuracy))

    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel")

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer,  rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only


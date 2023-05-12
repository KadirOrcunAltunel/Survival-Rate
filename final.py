from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, OneVsRest, MultilayerPerceptronClassifier

spark = SparkSession.builder.appName('birth').getOrCreate()
birth = spark.read.csv('/Users/kadiraltunel/Documents/births_transformed.csv', header=True, inferSchema=True)
birth.show(5)

birth.printSchema()

vectorAssembler = VectorAssembler(inputCols=["BIRTH_PLACE", "MOTHER_AGE_YEARS", "FATHER_COMBINED_AGE", "CIG_BEFORE",
                                             "CIG_1_TRI", "CIG_2_TRI", "CIG_3_TRI", "MOTHER_HEIGHT_IN",
                                             "MOTHER_PRE_WEIGHT", "MOTHER_DELIVERY_WEIGHT",
                                             "MOTHER_WEIGHT_GAIN", "DIABETES_PRE", "DIABETES_GEST", "HYP_TENS_GEST",
                                             "PREV_BIRTH_PRETERM"], outputCol="features")

birth_assembled = vectorAssembler.transform(birth)
birth_assembled.show(5)

indexer = StringIndexer(inputCol='INFANT_ALIVE_AT_REPORT', outputCol='label')
birth_indexed = indexer.fit(birth_assembled).transform(birth_assembled)
birth_indexed.show(5)

birth_indexed.select('INFANT_ALIVE_AT_REPORT', 'label').groupBy('INFANT_ALIVE_AT_REPORT', 'label').count().show()

birth_split = birth_indexed.randomSplit([0.8, 0.2], seed=1)
birth_train = birth_split[0]
birth_test = birth_split[1]

# Decision Tree

birth_tree = DecisionTreeClassifier()
birth_tree_model = birth_tree.fit(birth_train)

birth_prediction = birth_tree_model.transform(birth_test)

birth_prediction.select('label', 'features', 'prediction', 'probability').show(5)

evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator_accuracy.evaluate(birth_prediction)
print('Accuracy = %g ' % accuracy)
print('Test Error = %g ' % (1.0 - accuracy))

evaluator_precision = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='weightedPrecision')
weightedPrecision = evaluator_precision.evaluate(birth_prediction)
print('Precision = %g ' % weightedPrecision)

evaluator_recall = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='weightedRecall')
weightedRecall = evaluator_recall.evaluate(birth_prediction)
print('Recall = %g ' % weightedRecall)

# Logistic Regression

birth_log = LogisticRegression(featuresCol='features', labelCol='label', maxIter=5)
birth_log_model = birth_log.fit(birth_train)
birth_log_prediction = birth_log_model.transform(birth_test)

birth_log_prediction.select('label', 'features', 'prediction', 'probability').show(5)

log_evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy')
log_accuracy = log_evaluator_accuracy.evaluate(birth_log_prediction)
print('Accuracy = %g ' % log_accuracy)
print('Test Error = %g ' % (1.0 - log_accuracy))

log_evaluator_precision = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='weightedPrecision')
log_weightedPrecision = log_evaluator_precision.evaluate(birth_log_prediction)
print('Precision = %g ' % log_weightedPrecision)

log_evaluator_recall = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='weightedRecall')
log_weightedRecall = log_evaluator_recall.evaluate(birth_log_prediction)
print('Recall = %g ' % log_weightedRecall)

# SVM

base_log_birth = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

ovr = OneVsRest(classifier=base_log_birth)
ovrModel = ovr.fit(birth_train)

ovr_prediction = ovrModel.transform(birth_test)

ovr_evaluator_accuracy = MulticlassClassificationEvaluator(metricName='accuracy')
ovr_accuracy = ovr_evaluator_accuracy.evaluate(ovr_prediction)
print('Accuracy = %g ' % ovr_accuracy)
print('Test Error = %g ' % (1.0 - ovr_accuracy))

ovr_evaluator_precision = MulticlassClassificationEvaluator(metricName='weightedPrecision')
ovr_weightedPrecision = ovr_evaluator_precision.evaluate(ovr_prediction)
print('Precision = %g ' % ovr_weightedPrecision)

ovr_evaluator_recall = MulticlassClassificationEvaluator(metricName='weightedRecall')
ovr_weightedRecall = ovr_evaluator_recall.evaluate(ovr_prediction)
print('Recall = %g ' % ovr_weightedRecall)

# MLP

birth_layers = [birth_train.schema["features"].metadata["ml_attr"]["num_attrs"], 20, 10, 2]
birth_mlp = MultilayerPerceptronClassifier(layers=birth_layers, seed=1)
birth_mlp_model = birth_mlp.fit(birth_train)

birth_mlp_prediction = birth_mlp_model.transform(birth_test)

birth_evaluator_accuracy = MulticlassClassificationEvaluator(metricName='accuracy')
birth_accuracy = birth_evaluator_accuracy.evaluate(birth_mlp_prediction)
print('Accuracy = %g ' % birth_accuracy)
print('Test Error = %g ' % (1.0 - birth_accuracy))

birth_evaluator_precision = MulticlassClassificationEvaluator(metricName='weightedPrecision')
birth_weightedPrecision = birth_evaluator_precision.evaluate(birth_mlp_prediction)
print('Precision = %g ' % birth_weightedPrecision)

birth_evaluator_recall = MulticlassClassificationEvaluator(metricName='weightedRecall')
birth_weightedRecall = birth_evaluator_recall.evaluate(birth_mlp_prediction)
print('Recall = %g ' % ovr_weightedRecall)

# The best accuracy for model selection is with Decision Tree (0.72). Multilayer Perception is very close at (0.71)

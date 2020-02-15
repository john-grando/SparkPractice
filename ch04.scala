// Load data
val dataWithoutHeader = spark.read.
  option("inferSchema", true).
  option("header", false).
  csv("covtype/covtype.data")

// Add column names
val colNames = Seq(
  "Elevation", "Aspect", "Slope",
  "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
  "Horizontal_Distance_To_Roadways",
  "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
  "Horizontal_Distance_To_Fire_Points"
) ++ (
  (0 until 4).map(i => s"Wilderness_Area_$i")
) ++ (
  (0 until 40).map(i => s"Soil_Type_$i")
) ++ Seq("Cover_Type")

val data = dataWithoutHeader.toDF(colNames:_*).
  withColumn("Cover_Type", $"Cover_Type".cast("double"))

// Print data with header
data.head

// Split for train and test
val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
trainData.cache()
testData.cache()

// Arrange data into one column of vectors for MLib
import org.apache.spark.ml.feature.VectorAssembler
val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("featureVector")
val assembledTrainData = assembler.transform(trainData)

assembledTrainData.select("featureVector").show(truncate = false)

// Build classifier
import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random
val classifier = new DecisionTreeClassifier().
  setSeed(Random.nextLong()).
  setLabelCol("Cover_Type").
  setFeaturesCol("featureVector").
  setPredictionCol("prediction")
val model = classifier.fit(assembledTrainData)

// Print model decision tree
println(model.toDebugString)

// Print feature importance
model.featureImportances.toArray.zip(inputCols).
  sorted.reverse.foreach(println)

// Create dataframe with classication, prediction and probability array
val predictions = model.transform(assembledTrainData)

predictions.select("Cover_Type", "prediction", "probability").
  show(truncate = false)

// Perform multiclass evaluation of predictions
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("Cover_Type").
  setPredictionCol("prediction")

evaluator.setMetricName("accuracy").evaluate(predictions)
evaluator.setMetricName("f1").evaluate(predictions)

// Building confustion matrix; however, since this is built in the
// older API, it needs to be converted to data set (using as.[()])
// and then into an rdd (using rdd)
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val predictionRDD = predictions.
  select("prediction", "Cover_Type").
  as[(Double,Double)].
  rdd
val multiclassMetrics = new MulticlassMetrics(predictionRDD)
multiclassMetrics.confusionMatrix

// Confusion matrix directly using the DataFrame API
val confusionMatrix = predictions.
groupBy("Cover_Type").
  pivot("prediction", (1 to 7)).
  count().
  na.fill(0.0).
  orderBy("Cover_Type")
confusionMatrix.show()

// Calculate clas probabilities to create baseline accuracy
import org.apache.spark.sql.DataFrame
def classProbabilities(data: DataFrame): Array[Double] = {
  val total = data.count()
  data.groupBy("Cover_Type").count().
    orderBy("Cover_Type").
    select("count").as[Double].
    map(_ / total).
    collect()
}

val trainPriorProbabilities = classProbabilities(trainData)
val testPriorProbabilities = classProbabilities(testData)

trainPriorProbabilities.zip(testPriorProbabilities).map {
  case (trainProb, cvProb) => trainProb * cvProb
}.sum

// Create pipeline for combinations of hyperparameters to be tested
import org.apache.spark.ml.Pipeline
val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().
  setInputCols(inputCols).
  setOutputCol("featureVector")
val classifier = new DecisionTreeClassifier().
  setSeed(Random.nextLong()).
  setLabelCol("Cover_Type").
  setFeaturesCol("featureVector").
  setPredictionCol("prediction")
val pipeline = new Pipeline().setStages(Array(assembler, classifier))

// Use ParamGridBuilder to set up testing grid
import org.apache.spark.ml.tuning.ParamGridBuilder
val paramGrid = new ParamGridBuilder().
  addGrid(classifier.impurity, Seq("gini", "entropy")).
  addGrid(classifier.maxDepth, Seq(1, 20)).
  addGrid(classifier.maxBins, Seq(40, 300)).
  addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
  build()

// Create Evaluator to define label and metric used
val multiclassEval = new MulticlassClassificationEvaluator().
  setLabelCol("Cover_Type").
  setPredictionCol("prediction").
  setMetricName("accuracy")

// Split data to train/validation sets and run hyper parameter training
import org.apache.spark.ml.tuning.TrainValidationSplit
val validator = new TrainValidationSplit().
  setSeed(Random.nextLong()).
  setEstimator(pipeline).
  setEvaluator(multiclassEval).
  setEstimatorParamMaps(paramGrid).
  setTrainRatio(0.9)

  // Repartition trainData due to its size and available resrouces
  //val trainDataPartitioned = trainData.repartition(100)

//val validatorModel = validator.fit(trainDataPartitioned)

// Print best model  Commented because grid search also commented out to
// save time

// import org.apache.spark.ml.PipelineModel
// val bestModel = validatorModel.bestModel
// bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap

// Ouptut printed here so grid doesn't have to be run again
//{
//	dtc_066265adec3f-cacheNodeIds: false,
//	dtc_066265adec3f-checkpointInterval: 10,
//	dtc_066265adec3f-featuresCol: featureVector,
//	dtc_066265adec3f-impurity: entropy,
//	dtc_066265adec3f-labelCol: Cover_Type,
//	dtc_066265adec3f-maxBins: 300,
//	dtc_066265adec3f-maxDepth: 20,
//	dtc_066265adec3f-maxMemoryInMB: 256,
//	dtc_066265adec3f-minInfoGain: 0.0,
//	dtc_066265adec3f-minInstancesPerNode: 1,
//	dtc_066265adec3f-predictionCol: prediction,
//	dtc_066265adec3f-probabilityCol: probability,
//	dtc_066265adec3f-rawPredictionCol: rawPrediction,
//	dtc_066265adec3f-seed: -8253560235143534512
//}

// Show results from all models
//val validatorModel = validator.fit(trainData)
//val paramsAndMetrics = validatorModel.validationMetrics.
//  zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)
//paramsAndMetrics.foreach { case (metric, params) =>
//    println(metric)
//    println(params)
//    println()
//}

// Run best model on test test and compare accuracy
//validatorModel.validationMetrics.max
//multiclassEval.evaluate(bestModel.transform(testData))

// Make model based on grid results so it doesn't have to be run each time

val bestClassifier = new DecisionTreeClassifier().
  setSeed(Random.nextLong()).
  setLabelCol("Cover_Type").
  setFeaturesCol("featureVector").
  setPredictionCol("prediction").
  setImpurity("entropy").
  setMaxDepth(20).
  setMaxBins(300).
  setMinInfoGain(0.0)

val bestPipeline = new Pipeline().setStages(Array(assembler, bestClassifier))
val bestModelShort = bestPipeline.fit(trainData)

// Show train and test data metrics
multiclassEval.evaluate(bestModelShort.transform(trainData))
multiclassEval.evaluate(bestModelShort.transform(testData))

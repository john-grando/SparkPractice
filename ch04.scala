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

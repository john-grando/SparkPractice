// identify data files

// Full file
// val dataFile: String = "kddcup/kddcup.data"

// Sample file
val dataFile: String = "kddcup/kddcup.data_10_percent"

// Load data
val dataWithoutHeader = spark.read.
  option("inferSchema", true).
  option("header", false).
  csv(dataFile)

// Make DataFrame.
// Use if/else to partition data if it is large
// Hopefully this will prevent any max heap errors
val numPartitions: Int = if(dataWithoutHeader.count > 500000) {
  200
} else {
  sc.defaultMinPartitions
}

val data = dataWithoutHeader.toDF(
  "duration", "protocol_type", "service", "flag",
  "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
  "hot", "num_failed_logins", "logged_in", "num_compromised",
  "root_shell", "su_attempted", "num_root", "num_file_creations",
  "num_shells", "num_access_files", "num_outbound_cmds",
  "is_host_login", "is_guest_login", "count", "srv_count",
  "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
  "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
  "dst_host_count", "dst_host_srv_count",
  "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
  "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
  "dst_host_serror_rate", "dst_host_srv_serror_rate",
  "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
  "label").
  repartition(numPartitions)

// Investigate data labels
data.
  select("label").
  groupBy("label").
  count().
  orderBy($"count".desc).
  show(25)

// Create preliminary kmeans model
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
val numericOnly = data.drop("protocol_type", "service", "flag").cache()
val assembler = new VectorAssembler().
  setInputCols(numericOnly.columns.filter(_ != "label")).
  setOutputCol("featureVector")
val kmeans = new KMeans().
  setPredictionCol("cluster").
  setFeaturesCol("featureVector")
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val pipelineModel = pipeline.fit(numericOnly)
val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
kmeansModel.clusterCenters.foreach(println)

// Examine which labels are in which clusters
val withCluster = pipelineModel.transform(numericOnly)

withCluster.select("cluster", "label").
  groupBy("cluster", "label").count().
  orderBy($"cluster", $"count".desc).
  show(25)

// Create pipeline and run hyperparameter test for kmeans
import scala.util.Random
import org.apache.spark.sql.DataFrame
def clusteringScore0(data: DataFrame, k: Int): Double = {
  val assembler = new VectorAssembler().
    setInputCols(data.columns.filter(_ != "label")).
    setOutputCol("featureVector")
  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setPredictionCol("cluster").
    setFeaturesCol("featureVector")
  val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
  val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

// Run and Output results
(20 to 100 by 20).map(k => (k, clusteringScore0(numericOnly, k))).
foreach(println)

// Do it again with better controls
def clusteringScore1(data: DataFrame, k: Int): Double = {
  val assembler = new VectorAssembler().
    setInputCols(data.columns.filter(_ != "label")).
    setOutputCol("featureVector")
  val kmeans = new KMeans().
    setSeed(Random.nextLong()).
    setK(k).
    setMaxIter(40).
    setTol(1.0e-5).
    setPredictionCol("cluster").
    setFeaturesCol("featureVector")
  val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
  val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.computeCost(assembler.transform(data)) / data.count()
}

// Run and Output results
(20 to 100 by 20).map(k => (k, clusteringScore1(numericOnly, k))).
foreach(println)

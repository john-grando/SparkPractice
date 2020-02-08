// Load data
val rawUserArtistData =
  spark.
  read.
  textFile("msd/user_artist_data_small.txt")

// show data
rawUserArtistData.
  take(5).
  foreach(println)

// Create DataFrame. _* is to capture and discard all remaining items
val userArtistDF = rawUserArtistData.map { line =>
  val Array(user, artist, _*) = line.split(' ')
  (user.toInt, artist.toInt)
}.toDF("user", "artist")

// Compute stats
userArtistDF.agg(
min("user"), max("user"), min("artist"), max("artist")).show()

// Read artist data
val rawArtistData = spark.
  read.
  textFile("msd/artist_data_small.txt")

// Try to parse file, fails on bigger Dataset
// due to corrupted lines
rawArtistData.map { line =>
  val (id, name) = line.span(_ != '\t')
  (id.toInt, name.trim)
}.count()

// Better method of parsing file using flatMap
// because it doesn't have to have one output for
// every input.  It can have one, zero, or many
// outputs
val artistByID = rawArtistData.flatMap { line =>
  val (id, name) = line.span(_ != '\t')
    if (name.isEmpty) {
      None
    } else {
      try {
        Some((id.toInt, name.trim))
      } catch {
        case _: NumberFormatException => None
      }
    }
}.toDF("id", "name")

// Load artist aliases
val rawArtistAlias = spark.read.textFile("msd/artist_alias_small.txt")
  val artistAlias = rawArtistAlias.flatMap { line =>
    val Array(artist, alias) = line.split('\t')
    if (artist.isEmpty) {
      None
    } else {
      Some((artist.toInt, alias.toInt))
    }
}.collect().toMap

// show head
artistAlias.head

// show artist by id from map
artistByID.filter($"id" isin (1039896, 1277013)).show()

// Create helper function to map
// artist id to corrected artist id
// and return DataFrame for training
// data.
import org.apache.spark.sql._
import org.apache.spark.broadcast._
def buildCounts(
  rawUserArtistData: Dataset[String],
  bArtistAlias: Broadcast[Map[Int,Int]]
): DataFrame = {
  rawUserArtistData.map { line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    (userID, finalArtistID, count)
  }.toDF("user", "artist", "count")
}

// broadcast copies data to each executor instead of every
// task location.
val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
val trainData = buildCounts(rawUserArtistData, bArtistAlias)

// Cache DataFrame since the model will
// reference it mulitple times.
// This avoids re-computation of the DF
trainData.cache()

// Build the model
import org.apache.spark.ml.recommendation._
import scala.util.Random
val model = new ALS().
  setSeed(Random.nextLong()).
  setImplicitPrefs(true).
  setRank(10).
  setRegParam(0.01).
  setAlpha(1.0).
  setMaxIter(5).
  setUserCol("user").
  setItemCol("artist").
  setRatingCol("count").
  setPredictionCol("prediction").
  fit(trainData)

// Show feature vectors
model.userFactors.show(1, truncate = false)

// Spot check recoomendations
// Pick a userID
val userID = 1059637

val existingArtistIDs = trainData.
  filter($"user" === userID).
  select("artist").as[Int].collect()

// Show artists listened to by user
artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

// Make recommendations for artists and show top results
def makeRecommendations(
  model: ALSModel,
  userID: Int,
  howMany: Int
): DataFrame = {
  val toRecommend = model.itemFactors.
    select($"id".as("artist")).
    withColumn("user", lit(userID))

  model.transform(toRecommend).
    select("artist", "prediction").
    orderBy($"prediction".desc).
    limit(howMany)
}

// Need to enable cross-join to view table
spark.conf.set("spark.sql.crossJoin.enabled", "true")

// Get top recommendations and Show
val topRecommendations = makeRecommendations(model, userID, 5)
topRecommendations.show()

// Look up artist names from top topRecommendations
val recommendedArtistIDs =
  topRecommendations.select("artist").as[Int].collect()

artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

// load area under curve function from github site
:load ch03_function_areaUnderCurve.scala

// Get formatted DataFrame and make a train/cv split
val allData = buildCounts(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
trainData.cache()
cvData.cache()

// Get distinct set of artsit Ids, collect them to an array
// and make it into a broadcast variable
val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

// Train model
val model = new ALS().
  setSeed(Random.nextLong()).
  setImplicitPrefs(true).
  setRank(10).setRegParam(0.01).setAlpha(1.0).setMaxIter(5).
  setUserCol("user").setItemCol("artist").
  setRatingCol("count").setPredictionCol("prediction").
  fit(trainData)

// Compute mean AUC using CV data.
areaUnderCurve(cvData, bAllArtistIDs, model.transform)

// Create simple prediction algorithm as ground truth
// Use the globally most played artists as the prediction
// functions with two lists of arguments, by supplying one
// list, we create a partially applied function that returns
// another function which takes the second list as the argument.
def predictMostListened(train: DataFrame)(allData: DataFrame) = {
  val listenCounts = train.
    groupBy("artist").
    agg(sum("count").as("prediction")).
    select("artist", "prediction")

  allData.
    join(listenCounts, Seq("artist"), "left_outer").
    select("user", "artist", "prediction")
}

areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))

// train hyperparameters
val evaluations =
  for (rank <- Seq(5,30);
    regParam <- Seq(4.0, 0.0001);
    alpha <- Seq(1.0, 40.0))
  yield {
    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(rank).setRegParam(regParam).
      setAlpha(alpha).setMaxIter(20).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(trainData)

// Get AUC for each run
val auc = areaUnderCurve(cvData, bAllArtistIDs, model.transform)
  model.userFactors.unpersist()
  model.itemFactors.unpersist()
  (auc, (rank, regParam, alpha))
}

// Show auc results
evaluations.sorted.reverse.foreach(println)

// create new model based on hyperparameters
val modelPostHyper = new ALS().
  setSeed(Random.nextLong()).
  setImplicitPrefs(true).
  setRank(30).
  setRegParam(4).
  setAlpha(1.0).
  setMaxIter(5).
  setUserCol("user").
  setItemCol("artist").
  setRatingCol("count").
  setPredictionCol("prediction").
  fit(trainData)

// Show recommendations for user again
val topRecommendationsPostHyper = makeRecommendations(modelPostHyper, userID, 5)
topRecommendationsPostHyper.show()

val recommendedArtistIDsPostHyper =
  topRecommendationsPostHyper.select("artist").as[Int].collect()

artistByID.filter($"id" isin (recommendedArtistIDsPostHyper:_*)).show()

// Make recommendations for top 100
val someUsers = allData.select("user").as[Int].distinct().take(100)
val someRecommendations =
  someUsers.map(userID => (userID, makeRecommendations(modelPostHyper, userID, 5)))
someRecommendations.foreach { case (userID, recsDF) =>
  val recommendedArtists = recsDF.select("artist").as[Int].collect()
  println(s"$userID -> ${recommendedArtists.mkString(", ")}")
}

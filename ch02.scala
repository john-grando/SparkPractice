val parsed = spark.read.
  option("header", "true").
  option("nullValue", "?").
  option("inferSchema", "true").
  csv("linkage")

parsed.cache()
parsed.printSchema()

// RDD style of accessing data
parsed.rdd.
  map(_.getAs[Boolean]("is_match")).
  countByValue()

// DataFrame API style of accessing data
parsed.
  groupBy("is_match").
  count().
  orderBy($"count".desc).
  show()

// Create table so SQL syntax can be used to parse data
parsed.createOrReplaceTempView("linkage")

// SQL sytnax
spark.sql("""
  SELECT is_match, COUNT(*) cnt
  FROM linkage
  GROUP BY is_match
  ORDER BY cnt DESC
  """).show()

// Summary statistics
val summary = parsed.describe()

summary.show()
summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()

// Get matches and misses
// SQL style
val matches = parsed.where("is_match = true")
val matchSummary = matches.describe()
//DataFrame API style
val misses = parsed.filter($"is_match" === false)
val missSummary = misses.describe()

// load Pivot sript
:load Pivot.scala

// Create variables for matches and miss_desc
val matchSummaryT = pivotSummary(matchSummary)
val missSummaryT = pivotSummary(missSummary)

// Test join
matchSummaryT.createOrReplaceTempView("match_desc")
missSummaryT.createOrReplaceTempView("miss_desc")
spark.sql("""
  SELECT a.field, a.count + b.count total, a.mean - b.mean delta
  FROM match_desc a INNER JOIN miss_desc b ON a.field = b.field
  WHERE a.field NOT IN ("id_1", "id_2")
  ORDER BY delta DESC, total DESC
""").show()

// Create case class
case class MatchData(
  id_1: Int,
  id_2: Int,
  cmp_fname_c1: Option[Double],
  cmp_fname_c2: Option[Double],
  cmp_lname_c1: Option[Double],
  cmp_lname_c2: Option[Double],
  cmp_sex: Option[Int],
  cmp_bd: Option[Int],
  cmp_bm: Option[Int],
  cmp_by: Option[Int],
  cmp_plz: Option[Int],
  is_match: Boolean
)

// Create schema, not in book (workaround)
import org.apache.spark.sql.types.StructType
val matchSchema = new StructType().
  add("id_1", "int").
  add("id_2", "int").
  add("cmp_fname_c1", "double").
  add("cmp_fname_c2", "double").
  add("cmp_lname_c1", "double").
  add("cmp_lname_c2", "double").
  add("cmp_sex", "int").
  add("cmp_bd", "int").
  add("cmp_bm", "int").
  add("cmp_by", "int").
  add("cmp_plz", "int").
  add("is_match", "boolean")

// Reload data with schema
val parsed = spark.read.
  option("header", "true").
  option("nullValue", "?").
  schema(matchSchema).
  csv("linkage")

// Convert DataFrame to Dataset
val matchData = parsed.as[MatchData]
matchData.show()

import spark.implicits._
// Create score class
case class Score(value: Double) {
  def +(oi: Option[Int]) = {
    Score(value + oi.getOrElse(0))
  }
}

// Create scoring function
def scoreMatchData(md: MatchData): Double = {
  (Score(md.cmp_lname_c1.getOrElse(0.0)) +
  md.cmp_plz + md.cmp_by + md.cmp_bd +
  md.cmp_bm).value
}

// Create scored variable
val scored = matchData.map { md =>
  (scoreMatchData(md), md.is_match)
}.toDF("score", "is_match")

// Create contingency table function
def crossTabs(scored: DataFrame, t: Double): DataFrame = {
  scored.
  selectExpr(s"score >= $t as above", "is_match").
  groupBy("above").
  pivot("is_match", Seq("true", "false")).
  count()
}

// Test
crossTabs(scored, 4.0).show()
crossTabs(scored, 2.0).show()

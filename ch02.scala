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




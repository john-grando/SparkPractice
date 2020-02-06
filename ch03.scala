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

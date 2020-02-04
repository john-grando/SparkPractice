// Load data
val rawUserArtistData =
  spark.
  read.
  textFile("msd/user_artist_data_small.txt")

// show data
rawUserArtistData.
  take(5).
  foreach(println)

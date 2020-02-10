val dataWithoutHeader = spark.read.
  option("inferSchema", true).
  option("header", false).
  csv("covtype/covtype.data")

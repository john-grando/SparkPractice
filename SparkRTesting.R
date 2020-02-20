#Setup and start session
Sys.setenv(SPARK_HOME = "/usr/local/spark")
Sys.setenv(JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]",
               sparkConfig = list(spark.driver.memory = "4g"))


#Create dataframe
clusters_data <- read.df("kddcup/kddcup.data_10_percent", "csv",
                         inferSchema = "true", header = "false")
colnames(clusters_data) <- c(
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
  "label")

#Only numeric columns
numeric_only <- cache(drop(clusters_data,
                           c("protocol_type", "service", "flag", "label")))
#Create model
kmeans_model <- spark.kmeans(numeric_only, ~ .,
                             k = 100, maxIter = 40, initMode = "k-means||")

#make predictions and get sample
clustering <- predict(kmeans_model, numeric_only)
clustering_sample <- SparkR::collect(sample(clustering, FALSE, 0.01))
str(clustering_sample)

#Evaluate clusters
clusters <- clustering_sample["prediction"]
data <- data.matrix(within(clustering_sample, rm("prediction")))
table(clusters)

#project down to 3 dimensional data and plot
library(rgl)
random_projection <- matrix(data = rnorm(3*ncol(data)), ncol = 3)
random_projection_norm <-
  random_projection / sqrt(rowSums(random_projection*random_projection))
projected_data <- data.frame(data %*% random_projection_norm)

num_clusters <- max(clusters)
palette <- rainbow(num_clusters)
colors = sapply(clusters, function(c) palette[c])
plot3d(projected_data, col = colors, size = 10)

#do with pca
library(dplyr)
data_df <- as.data.frame(data)
bad_cols <- (colnames(data_df[, sapply(data_df, function(col) length(unique(col)))<=1]))
data.pca <- prcomp(data_df %>% select(-bad_cols), center = TRUE, scale. = TRUE)
summary(data.pca)
plot3d(data.pca$x[,c(1,2,3)], col = colors, size = 10)

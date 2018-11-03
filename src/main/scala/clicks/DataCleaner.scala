/*import org.apache.spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

val spark = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()

val data = spark.read.json("/home/godefroi/Téléchargements/data-students-new.json")

val labelAsInt = data.withColumn("label",$"label".cast("Int"))

val cleanBidFloor = labelAsInt.na.fill(0, Seq("bidFloor"))

val cleanNullInterests = cleanBidFloor.na.fill("NC", Seq("interests"))

val cleanSize = cleanNullInterests.filter("size is not NULL").select(
    col("appOrSite"),
    col("bidFloor"),
    col("interests"),
    col("label"),
    col("media"),
    col("publisher"),
    col("size"),
    col("timestamp"),
    col("user")
)

def cleanInterests(col: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    regexp_replace(col, "-[0-9]", "")
}

val renameInterests = cleanSize.withColumn("interests", cleanInterests(col("interests")))

val sizeToString = renameInterests.withColumn("size", concat_ws("x", $"size"))

// cleanSize.coalesce(1).write.json("/Users/johan/Downloads/data-students-new1.json")
sizeToString.coalesce(1).write.json("/home/godefroi/Téléchargements/new2JSON.json")
*/
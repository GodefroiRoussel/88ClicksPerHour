val data = spark.read.json("/Users/Marion/Desktop/data-students.json")

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
sizeToString.coalesce(1).write.json("/Users/Marion/Desktop/data-students-new2")
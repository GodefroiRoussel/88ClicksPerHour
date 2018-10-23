val data = spark.read.json("/Users/johan/Downloads/data-students.json")
val cleanBidFloor = data.na.fill(0, Seq("bidFloor"))

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

// cleanSize.coalesce(1).write.json("/Users/johan/Downloads/data-students-new1.json")
renameInterests.coalesce(1).write.json("/Users/johan/Downloads/data-students-new2")
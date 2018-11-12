package clicks

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.Column
import org.apache.spark.sql._


object DataCleaner extends App {
  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val data = spark.read.json("/Users/assilelyahyaoui/Documents/data-students.json")

  def booleanToInt( bool :Boolean) = if(bool) 1 else 0
  val booleanToInt_column = udf(booleanToInt _)

  /**
    * transform label attribute to int
    *
    * @param dataFrame DataFrame to change
    */
  def label(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("label", booleanToInt_column(dataFrame("label")))
  }

  def cleanBidFloor(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill(0,Seq("bidFloor"))
  }

  def cleanNullInterests(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("NC",Seq("interests"))
  }


  def cleanSize(dataFrame: DataFrame):DataFrame ={
    dataFrame.filter("size is not NULL").select(
      col("appOrSite"),
      col("bidFloor"),
      col("interests"),
      col("label"),
      col("media"),
      col("publisher"),
      col("size"),
      col("user")
    )

  }

  def cleanInterestsRegex(col: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    regexp_replace(col, "-[0-9]", "")
  }


  def cleanInterests(dataFrame: DataFrame) : DataFrame={
    dataFrame.withColumn("interests", cleanInterestsRegex(col("interests")))
  }

  def sizeToString(dataFrame: DataFrame): DataFrame={
    dataFrame.withColumn("size", concat_ws("x", $"size"))
  }

  def newDf(dataFrame: DataFrame): DataFrame ={
    var ndf = dataFrame
    ndf = label(ndf)
    ndf = cleanBidFloor(ndf)
    ndf = cleanNullInterests(ndf)
    ndf = cleanSize(ndf)
    ndf = cleanInterests(ndf)
    ndf = sizeToString(ndf)
    return ndf
  }

  //sizeToString.coalesce(1).write.json("/Users/assilelyahyaoui/Documents/data-students-cleaned.json")
  // cleanSize.coalesce(1).write.json("/Users/johan/Downloads/data-students-new1.json")

}
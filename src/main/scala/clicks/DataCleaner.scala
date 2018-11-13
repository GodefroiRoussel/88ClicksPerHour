package clicks

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.sql.Column
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{concat, lit}

object DataCleaner {
  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  //val data = spark.read.json("/home/godefroi/Téléchargementsdata-students.json")

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
      val columnNames = Seq("appOrSite", "bidFloor", "interests", "label", "media", "publisher", "user")

      //val df1 : DataFrame = dataFrame.filter("size is not NULL")
      dataFrame.select( columnNames.head, columnNames.tail: _*)
  }

  def cleanInterestsRegex(col: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    regexp_replace(col, "-[0-9]", "")
  }


  def cleanInterests(dataFrame: DataFrame) : DataFrame={
    dataFrame.withColumn("interests", cleanInterestsRegex(dataFrame("interests")))
  }

  def sizeToString(dataFrame: DataFrame): DataFrame={
      val colSize: Column = dataFrame("size")
      val test : String = colSize.toString()
      println(test)
    //dataFrame.withColumn("size", concat("x", "size"))
      dataFrame
  }

  def newDf(dataFrame: DataFrame): DataFrame ={
    var ndf = dataFrame
    ndf = label(ndf)
    ndf = cleanBidFloor(ndf)
    ndf = cleanNullInterests(ndf)
    ndf = cleanInterests(ndf)
      ndf = cleanSize(ndf)
    return ndf
  }

  //sizeToString.coalesce(1).write.json("/Users/assilelyahyaoui/Documents/data-students-cleaned.json")
  // cleanSize.coalesce(1).write.json("/Users/johan/Downloads/data-students-new1.json")

}
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

  def booleanToInt( bool :Boolean) = if(bool) 1.0 else 0.0
  val booleanToInt_column = udf(booleanToInt _)

  /**
    * transform label attribute to int
    *
    * @param dataFrame DataFrame to change
    */
  def label(dataFrame: DataFrame): DataFrame = {
    //dataFrame.withColumn("label", booleanToInt_column(dataFrame("label")))
    dataFrame.withColumn("label",dataFrame("label").cast("Int"))
  }

  def cleanBidFloor(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill(0,Seq("bidFloor"))
  }

  def cleanNullInterests(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("NC",Seq("interests"))
  }


  def castSize(dataFrame: DataFrame):DataFrame ={
    dataFrame.withColumn("size", dataFrame("size").cast("string"))
  }

  def cleanSize(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("UNKNOWN",Seq("size"))
  }


  def cleanType(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("UNKNOWN",Seq("type"))
  }

  def selectData(dataFrame: DataFrame, forPrediction: Boolean): DataFrame = {
    val columnNames = forPrediction match {
      case true => Seq("appOrSite", "bidFloor", "interests", "media", "publisher", "user", "size", "type")
      case _ => Seq("appOrSite", "bidFloor", "interests", "label", "media", "publisher", "user", "size", "type")
    }
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
   // dataFrame.withColumn("size", test)
     // dataFrame.withColumn("size", concat("x", "size"))
    //dataFrame.withColumn("size", colSize.)
      dataFrame
  }

  def newDf(dataFrame: DataFrame, forPrediction: Boolean): DataFrame ={
    var ndf = dataFrame
    if (!forPrediction)
      ndf = label(ndf)

    ndf = cleanBidFloor(ndf)
    ndf = cleanNullInterests(ndf)
    ndf = castSize(ndf)
    ndf = cleanSize(ndf)
    ndf = cleanType(ndf)
    ndf = selectData(ndf, forPrediction)
    ndf = cleanInterests(ndf)
    ndf = sizeToString(ndf)
    return ndf
  }

  //sizeToString.coalesce(1).write.json("/Users/assilelyahyaoui/Documents/data-students-cleaned.json")
  // cleanSize.coalesce(1).write.json("/Users/johan/Downloads/data-students-new1.json")

}
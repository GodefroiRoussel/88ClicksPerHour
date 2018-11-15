package clicks

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.functions.{concat, lit}
import org.apache.spark.sql.Column
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{concat, lit}

// Cleans the DataFrame according to our choices
object DataCleaner {

  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  /**
    * transform the label attribute to int
    *
    * @param dataFrame DataFrame to change
    */
  def label(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("label",dataFrame("label").cast("Int"))
  }

  /**
    * replace the null columns of bidFloor with 0
    *
    * @param dataFrame DataFrame to change
    */
  def cleanBidFloor(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill(0,Seq("bidFloor"))
  }

  /**
    * replace the null columns of interests with "N/A" for not applicable
    *
    * @param dataFrame DataFrame to change
    */
  def cleanNullInterests(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("N/A",Seq("interests"))
  }

  /**
    * transform the size attribute to string
    *
    * @param dataFrame DataFrame to change
    */
  def castSize(dataFrame: DataFrame):DataFrame ={
    dataFrame.withColumn("size", dataFrame("size").cast("string"))
  }

  /**
    * replace the null columns of size with "UNKNOWN" 
    *
    * @param dataFrame DataFrame to change
    */
  def cleanSize(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("UNKNOWN",Seq("size"))
  }

  /**
    * replace the null columns of type with "UNKNOWN" 
    *
    * @param dataFrame DataFrame to change
    */
  def cleanType(dataFrame: DataFrame) : DataFrame ={
    dataFrame.na.fill("UNKNOWN",Seq("type"))
  }

  /**
    * applies a regex expression to the column
    *
    * @param col Column to change
    */
  def cleanInterestsRegex(col: org.apache.spark.sql.Column): org.apache.spark.sql.Column = {
    regexp_replace(col, "-[0-9]", "")
  }

  /**
    * uses the cleanInterestsRegex to replace the interests columns to remove the sub interests
    *
    * @param dataFrame DataFrame to change
    */
  def cleanInterests(dataFrame: DataFrame) : DataFrame={
    dataFrame.withColumn("interests", cleanInterestsRegex(dataFrame("interests")))
  }

  /**
    * select only the columns of the dataFrame that we want to keep
    *
    * @param dataFrame DataFrame to change
    * @param forPrediction boolean, equals true if we use it for the prediction program, false otherwise
    */
  def selectData(dataFrame: DataFrame, forPrediction: Boolean): DataFrame = {
    val columnNames = forPrediction match {
      case true => Seq("appOrSite", "bidFloor", "interests", "media", "publisher", "user", "size", "type")
      case _ => Seq("appOrSite", "bidFloor", "interests", "label", "media", "publisher", "user", "size", "type")
    }
    dataFrame.select( columnNames.head, columnNames.tail: _*)
  }
  
    /**
    * applies all the functions to the dataFrame
    *
    * @param dataFrame DataFrame to change
    * @param forPrediction boolean, equals true if we use it for the prediction program, false otherwise
    */
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
    return ndf
  }
}
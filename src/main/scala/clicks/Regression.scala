package clicks

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.Pipeline

object ClickPrediction extends App {
  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.json("/home/godefroi/Téléchargements/data-students.json")

  val data = DataCleaner.newDf(df)

  def indexStringColumns(df : DataFrame, col: String) : DataFrame = {
    var newdf : DataFrame = df
    val si : StringIndexer = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")
    val sm: StringIndexerModel = si.fit(newdf)
    val indexed : DataFrame = sm.transform(newdf).drop(col)
    newdf = indexed

    return newdf
  }


  def indexStringColumns2(df : DataFrame, cols: Array[String]) : DataFrame ={
    var newdf = df

    for (col <- cols){
      val si = new StringIndexer().setInputCol(col).setOutputCol(col +"Index")

      val sm :  StringIndexerModel = si.fit(newdf)
      val indexed =  sm.transform(newdf).drop(col)
      newdf = indexed

    }
    return newdf
  }

  /*
  val appOrSiteIndexer: DataFrame = indexStringColumns(data, "appOrSite")
  val interestsIndexer: DataFrame = indexStringColumns(appOrSiteIndexer, "interests")
  val mediaIndexer: DataFrame = indexStringColumns(interestsIndexer, "media")
  val publisherIndexer: DataFrame = indexStringColumns(mediaIndexer, "publisher")
  //val sizeIndexer: DataFrame = indexStringColumns(publisherIndexer, "size")
  val userIndexer: DataFrame = indexStringColumns(publisherIndexer, "user")*/
  val userIndexer: DataFrame= indexStringColumns2(data, Array("appOrSite", "interests", "media", "publisher", "user"))

  println("-----------------------------         USER INDEXER               ------------------------------------------------------")
  userIndexer.printSchema()
  userIndexer.show()


  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("bidFloor", "appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "userIndex"))
    .setOutputCol("features")

  val testValue = 0.25
  val training = 0.75
  val splits : Array[DataFrame] = userIndexer.randomSplit(Array(training, testValue))
  var trainData : DataFrame = splits(0)
  var testData : DataFrame = splits(1)

  var train: DataFrame = assembler.transform(trainData)


  var test: DataFrame = assembler.transform(testData)


  /*train = train.select("features", "label")
  train.printSchema()
  train.select("features").show()
  test = test.select("features", "label")
  test.printSchema()
  test.select("features").show()*/


  // Train the model

  val lr: LogisticRegression = new LogisticRegression()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setRegParam(0.0)
    .setElasticNetParam(0.0)
    .setMaxIter(10)
    .setTol(1E-6)
    .setFitIntercept(true)

  val stages = Array(assembler,lr)

  val pipeline = new Pipeline().setStages(stages)

  //val model: LogisticRegressionModel = lr.fit(train)
  val model = pipeline.fit(train)
  //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


  // Test the model
  val predictions: DataFrame = model.transform(test)
  predictions.printSchema()
  predictions.show
  predictions.select ("label", "prediction","rawPrediction").show()
  predictions.select("prediction").distinct().show()
    predictions.select("label").distinct().show()
  val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
    .setLabelCol("label")

  val eval = evaluator.evaluate(predictions)
  println("Test set areaunderROC/accuracy = " + eval)

  // Exporte en csv
  //predictions.select("label", "prediction").write.format("csv").option("header","true").save("/Documents")

  spark.close()
}



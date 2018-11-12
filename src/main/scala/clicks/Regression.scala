package clicks

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.implicits._
import org.apache.spark.sql.{DataFrame, SparkSession}

object ClickPrediction extends App {
  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val df = spark.read.json("/home/godefroi/Téléchargements/new2JSON.json")

  val data = DataCleaner.newDf(df)

  def indexStringColumns(df : DataFrame, col: String) : DataFrame = {
    var newdf : DataFrame = df
    val si : StringIndexer = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")
    val sm: StringIndexerModel = si.fit(newdf)
    val indexed : DataFrame = sm.transform(newdf).drop(col)
    newdf = indexed

    return newdf
  }



  // One hot encoder estimator
  // Maps a column of label indices to a column of binary vectors, with at most a single one-value. That alows Logistic Regression to use categorical features
  def oneHot(df : DataFrame, cols: Array[String]) : DataFrame ={

    var newdf : DataFrame = df

    for (col <- cols){

      val oh : OneHotEncoderEstimator = new OneHotEncoderEstimator()
        .setInputCols(Array(col))
        .setOutputCols(Array(col + "Enc"))
      val model : OneHotEncoderModel = oh.fit(newdf)
      val encoded : DataFrame = model.transform(newdf).drop(col)
      newdf = encoded

    }
    return newdf

  }

  val appOrSiteIndexer: DataFrame = indexStringColumns(data, "appOrSite")
  val interestsIndexer: DataFrame = indexStringColumns(appOrSiteIndexer, "interests")
  val mediaIndexer: DataFrame = indexStringColumns(interestsIndexer, "media")
  val publisherIndexer: DataFrame = indexStringColumns(mediaIndexer, "publisher")
  val sizeIndexer: DataFrame = indexStringColumns(publisherIndexer, "size")
  val userIndexer: DataFrame = indexStringColumns(sizeIndexer, "user")

  println("-----------------------------         USER INDEXER               ------------------------------------------------------")
  userIndexer.printSchema()
  userIndexer.show()

  val encoder : DataFrame = oneHot(userIndexer, Array("appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex" , "userIndex"))
  println(" -----------------------------          ENCODER               ------------------------------------------------------")
  encoder.printSchema()
  encoder.show()

  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("bidFloor", "appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex", "userIndex"))
    .setOutputCol("features")

  val test = 0.25
  val training = 0.75
  val splits : Array[DataFrame] = userIndexer.randomSplit(Array(training, test))
  var trainData : DataFrame = splits(0)
  var testData : DataFrame = splits(1)

  var train: DataFrame = assembler.transform(trainData)
  train.printSchema()

  var test: DataFrame = assembler.transform(testData)
  test.printSchema()

  train = train.select("features", "label")
  test = testData.select("features", "label")


  // Train the model

  val lr: LogisticRegression = new LogisticRegression()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setRegParam(0.0)
    .setElasticNetParam(0.0)
    .setMaxIter(10)
    .setTol(1E-6)
    .setFitIntercept(true)

  val model: LogisticRegressionModel = lr.fit(train)
  println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


  // Test the model
  val predictions: DataFrame = model.transform(test)
  predictions.printSchema()
  predictions.show
  predictions.select ("label", "prediction","rawPrediction").show()

  /*val evaluator = new BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("rawPrediction")
    .setLabelCol("label")

  val eval = evaluator.evaluate(prediction)
  println("Test set areaunderROC/accuracy = " + eval)*/

  // Exporte en csv
  predictions.select("label", "prediction").write.format("csv").option("header","true").save("/Documents")

  spark.close()
}



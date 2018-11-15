package clicks

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}

object ClickPrediction extends App {

  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val df = spark.read.json("/Users/assilelyahyaoui/Documents/data-students.json")

  val data = DataCleaner.newDf(df)


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

  val userIndexer: DataFrame= indexStringColumns2(data, Array("appOrSite", "interests","media", "publisher", "user", "size", "type"))

  println("-----------------------------         USER INDEXER               ------------------------------------------------------")
  userIndexer.printSchema()
  userIndexer.show()

  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("label") === 0).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("label")))
    weightedDataset
  }

  val dfTest = balanceDataset(userIndexer)

  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("appOrSiteIndex", "interestsIndex","mediaIndex", "publisherIndex", "userIndex", "sizeIndex", "typeIndex"))
    .setOutputCol("features")

  val testValue = 0.2
  val training = 0.8
  val splits : Array[DataFrame] = dfTest.randomSplit(Array(training, testValue))
  var trainData : DataFrame = splits(0)
  var testData : DataFrame = splits(1)

  //var train: DataFrame = assembler.transform(trainData)

  //var test: DataFrame = assembler.transform(testData)


  /*train = train.select("features", "label")
  train.printSchema()
  train.select("features").show()
  test = test.select("features", "label")
  test.printSchema()
  test.select("features").show()*/


  // Train the model

  val lr: LogisticRegression = new LogisticRegression()
    .setWeightCol("classWeightCol")
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setMaxIter(10)

  val stages = Array(assembler,lr)

  val pipeline: Pipeline = new Pipeline().setStages(stages)

  //val model: LogisticRegressionModel = lr.fit(train)
  val model: PipelineModel = pipeline.fit(trainData)
  //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


  // Test the model
  val predictions: DataFrame = model.transform(testData)


  /*predictions.printSchema()
  predictions.show()
  predictions.select ("label", "prediction","rawPrediction").show()*/
  predictions.select("prediction").distinct().show()

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
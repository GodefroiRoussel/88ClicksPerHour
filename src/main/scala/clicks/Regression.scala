package clicks
import scala.io.StdIn

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}

object ClickPrediction extends App {

  def creationModel(): Unit = {
    val spark = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.json("/home/godefroi/Téléchargements/data-students.json")

    val data = DataCleaner.newDf(df, false)

    val userIndexer: DataFrame= indexStringColumns2(data, Array("appOrSite", "interests","media", "publisher", "user", "size", "type"))

    println("-----------------------------         USER INDEXER               ------------------------------------------------------")
    userIndexer.printSchema()
    userIndexer.show()

    val dfTest = balanceDataset(userIndexer)

    val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(Array("appOrSiteIndex", "interestsIndex","mediaIndex", "publisherIndex", "userIndex", "sizeIndex", "typeIndex"))
        .setOutputCol("features")

    val testValue = 0.2
    val training = 0.8
    val splits: Array[DataFrame] = dfTest.randomSplit(Array(training, testValue))
    var trainData: DataFrame = splits(0)
    var testData: DataFrame = splits(1)

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

    val stages = Array(assembler, lr)

    val pipeline: Pipeline = new Pipeline().setStages(stages)

    //val model: LogisticRegressionModel = lr.fit(train)
    val model: PipelineModel = pipeline.fit(trainData)
    //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
    println("Avant le model")
    // Saving model to the path and overwrite the file if already existing
    //model.write.overwrite().save("/home/godefroi/Téléchargements/modelSaved")

    println("Après le model")
    // Test the model
    val predictions: DataFrame = model.transform(testData)

    val lorModel = model.stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"LogisticRegression: ${(lorModel :LogisticRegressionModel)}")
    // Print the weights and intercept for logistic regression.
    println(s"Weights: ${lorModel.coefficients} Intercept: ${lorModel.intercept}")

    /*predictions.printSchema()
    predictions.show()
    predictions.select ("label", "prediction","rawPrediction").show()*/
    //predictions.select("prediction").distinct().show()
    //predictions.select("label").distinct().show()

    val evaluator = new BinaryClassificationEvaluator()
        .setMetricName("areaUnderROC")
        .setRawPredictionCol("rawPrediction")
        .setLabelCol("label")

    val eval = evaluator.evaluate(predictions)
    println("Test set areaunderROC/accuracy = " + eval)


    // Exporte en csv
    predictions.select("label", "prediction").write.format("csv").option("header", "true").save("/home/godefroi/Téléchargements/test.csv")

    spark.close()
  }

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


  def batch(filePath: String, modelPath: String): Unit = {
    val spark = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val input = StdIn.readLine()

    val df = spark.read.json(filePath)

    val data = DataCleaner.newDf(df, true)

    val model = PipelineModel.load(modelPath)

    println("J'ai bien chargé le model")



  }

  def chooseMode(): String = {
    println("Do you want to create a model or to predict some data from unlabelled data ?\n" +
        "Press 1 to create a model.\n" +
        "Press 2 to predict clicks from unlabelled data.\n" +
        "Press any other key to quit the program.\n")

    StdIn.readLine()
  }

  def end(): Unit = {

  }

  //-------------------- MAIN ------------------------
  creationModel()
  /*main()

  def main(): Unit ={
    val mode: String = chooseMode()
    mode match {
      case "1" => creationModel()
      case "2" => batch("", "")
      case _ =>     println("Good Bye")
        return
    }
  }*/


  // ------------------------- FUNCTIONS --------------------------
  def indexStringColumns2(df: DataFrame, cols: Array[String]): DataFrame = {
    var newdf = df

    for (col <- cols) {
      val si = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")

      val sm: StringIndexerModel = si.fit(newdf)
      val indexed = sm.transform(newdf).drop(col)
      newdf = indexed

    }
    return newdf
  }

}
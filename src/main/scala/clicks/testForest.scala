import clicks.ClickPrediction.{assembler, predictions, testData, trainData, _}
import clicks.DataCleaner
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

object test extends App {
  val spark = SparkSession
    .builder()
    .appName("Click prediction")
    .config("spark.master", "local")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val df = spark.read.json("/Users/assilelyahyaoui/Documents/data-students.json")
  val count = df.select("label").distinct().count()
  println("count up")
  val data = DataCleaner.newDf(df)
  data.select("label").distinct().show()

  println("-----------------------------         DATA CLEAN          ------------------------------------------------------")
  data.printSchema()
  data.show(50)
  data.select("label").distinct().show()

  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("label") === 0).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = { d: Double =>
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
  def indexStringColumns(df : DataFrame, col: String) : DataFrame = {
    var newdf : DataFrame = df
    val si : StringIndexer = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")
    val sm: StringIndexerModel = si.fit(newdf)
    val indexed : DataFrame = sm.transform(newdf).drop(col)
    newdf = indexed

    return newdf
  }

  /* val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("appOrSiteIndex")
   val interestsIndexer = new StringIndexer().setInputCol("interests").setOutputCol("interestsIndex")
   val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("mediaIndex")
   val publisherIndexer = new StringIndexer().setInputCol("publisher").setOutputCol("publisherIndex")
   val userIndexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex")

   val appOrSiteIndexer: DataFrame = indexStringColumns(data, "appOrSite")
   val interestsIndexer: DataFrame = indexStringColumns(appOrSiteIndexer, "interests")
   val mediaIndexer: DataFrame = indexStringColumns(interestsIndexer, "media")
   val publisherIndexer: DataFrame = indexStringColumns(mediaIndexer, "publisher")
   val userIndexer: DataFrame = indexStringColumns(publisherIndexer, "user")*/

  /*println("-----------------------------         USER INDEXER               ------------------------------------------------------")
  userIndexer.printSchema()
  userIndexer.show()
*/

  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("bidFloor"))
    .setOutputCol("features")

  val assembeleddf = assembler.transform(data)

  val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("appOrSiteIndex")
  val interestsIndexer = new StringIndexer().setInputCol("interests").setOutputCol("interestsIndex")
  val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("mediaIndex")
  val publisherIndexer = new StringIndexer().setInputCol("publisher").setOutputCol("publisherIndex")
  val userIndexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex")

  val appOrSitedf = appOrSiteIndexer.fit(assembeleddf).transform(assembeleddf)
  val interestsIndexerdf = interestsIndexer.fit(appOrSitedf).transform(appOrSitedf)
  val mediaIndexerdf = mediaIndexer.fit(interestsIndexerdf).transform(interestsIndexerdf)
  val publisherIndexerdf = publisherIndexer.fit(mediaIndexerdf).transform(mediaIndexerdf)
  val userIndexerdf = userIndexer.fit(publisherIndexerdf).transform(publisherIndexerdf)

  userIndexerdf.show()
  userIndexerdf.select("label").distinct().show()


  val testing = 1
  val training = 1
  val splitSeed = 5043

  val Array(trainingData, testingData) = userIndexerdf.randomSplit(Array(training, testing), splitSeed)
  trainingData.select("label").distinct().show()
  testingData.select("label").distinct().show()

  val classifier = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(3)
    .setNumTrees(20)
    .setFeatureSubsetStrategy("auto")
    .setSeed(5043)


  val model1 = classifier.fit(trainingData)
  val predictions1 = model1.transform(testingData)

  predictions1.show(50)
  predictions1.select("label").distinct().show()
  predictions1.select("prediction").distinct().show()

  /*
      val trainData : DataFrame = splits(0)
      val testData : DataFrame = splits(1)


      var train: DataFrame = assembler.transform(trainData)


      var test: DataFrame = assembler.transform(testData)


      val lr: LogisticRegression = new LogisticRegression()
          /*.setLabelCol("label")
           .setFeaturesCol("features")
          .setPredictionCol("prediction")
          .setMaxIter(10)
          .setRegParam(0.01)
  */
      val stages = Array(appOrSiteIndexer, interestsIndexer, mediaIndexer, publisherIndexer, userIndexer, assembler, lr)

      val pipeline = new Pipeline().setStages(stages)

     val model = pipeline.fit(train)

     //val model: LogisticRegressionModel = lr.fit(trainData)
     //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


      // Test the model
      val predictions: DataFrame = model.transform(test)
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

  */
  spark.close()
}
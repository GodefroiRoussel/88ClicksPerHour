package clicks
import scala.io.StdIn
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import scala.io.StdIn.readLine

object ClickPrediction extends App {


  //-------------------- MAIN ------------------------

    println("Please enter the path to your JSON file you want to predict")
    val filePath: String = readLine().trim()
    clickPrediction(filePath, "./modelSaved")



  //----------------------- CREATION MODEL --------------------------

    /**
    * allows to create the classification model
    *
    */
  def creationModel(): Unit = {

    val spark: SparkSession = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // The json file
    val df: DataFrame = spark.read.json("/home/godefroi/Téléchargements/data-students.json")

    // We clean the dataFrame with our dataCleaner program
    val data: DataFrame = DataCleaner.newDf(df, false)
    println("Data Cleaned")
    data.printSchema()

    // We index categorical values in order to do the logistic regression
    val userIndexer: DataFrame= indexStringColumns2(data, Array("appOrSite", "interests","media", "publisher", "user", "size", "type"))

    // We create a dataSet with a new colum classWeightCol that contains the weight of the labels 
    val dfBalanced: DataFrame = balanceDataset(userIndexer)

    // Transform the list of indexed columns into a single vector column.
    val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(Array("bidFloor", "appOrSiteIndex", "interestsIndex","mediaIndex", "publisherIndex", "userIndex", "sizeIndex", "typeIndex"))
        .setOutputCol("features")

    // We split our new dataset in two, one dataSet to train our model and one to test it
    val testValue: Double = 0.2
    val training: Double = 0.8
    val splits: Array[DataFrame] = dfBalanced.randomSplit(Array(training, testValue))
    var trainData: DataFrame = splits(0)
    var testData: DataFrame = splits(1)

    // We create our logistic regression. The column with the label's weight is the new one from balanceDataset, classWeightCol 
    // The colum to predict is the label and the columns to explain this label are the features
    // The setMaxIter make the logistic regression use at most 10 iterations
    val lr: LogisticRegression = new LogisticRegression()
        .setWeightCol("classWeightCol")
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setMaxIter(10)

    // A Pipeline is specified as a sequence of stages, and each stage is either a Transformer or an Estimator
    // A Transformer converts a DataFrame into another, by appending one or more columns. (Here, assembler)
    // An Estimator produces a Model from a DataFrame (Here, Logistic Regression)
    val stages = Array(assembler, lr)

    // The DataFrame passes through each stage and is transformed
    val pipeline: Pipeline = new Pipeline().setStages(stages)

    // We train the model
    val model: PipelineModel = pipeline.fit(trainData)

    // Saving model to the path and overwrite the file if already existing
    model.write.overwrite().save("./modelSaved")

    // We test the model
    val predictions: DataFrame = model.transform(testData)

    // We use the stage Estimator, LogisticRegression of the pipeline to print the coefficients
    val lorModel: LogisticRegressionModel = model.stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"LogisticRegression: ${(lorModel :LogisticRegressionModel)}")

    // Print the coefficients and intercept from our logistic regression
    println(s"Coefficients: ${lorModel.coefficients} Intercept: ${lorModel.intercept}")

    // We use an Evaluator to compute metrics that indicate how good our model is
    //BinaryClassificationEvaluator is use for binary classifications like our LogisticRegression
    val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
        .setMetricName("areaUnderROC")
        .setRawPredictionCol("rawPrediction")
        .setLabelCol("label")

    // We evaluate and print out metrics, like our model accuracy
    val eval: Double = evaluator.evaluate(predictions)
    println("Test set areaunderROC/accuracy = " + eval)

    spark.close()
  }


  //---------------------------------- CLICK PREDICTION -----------------------------------

    /**
    * applies our model to a file and predicts the number of clicks
    * 
    * @param filePath path to the Json with the data
    * @param modelPath path to our model
    */
  def clickPrediction(filePath: String, modelPath: String): Unit = {
    // Creating spark session and read the data from the JSON of the user
    val spark: SparkSession = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val initDf: DataFrame = spark.read.json(filePath)

    // We clean the dataFrame with our dataCleaner program
    val data: DataFrame = DataCleaner.newDf(initDf, true)

    // We load our model
    val model: PipelineModel = PipelineModel.load(modelPath)

    // We index categorical values in order to do the logistic regression according to our model
    val userIndexer: DataFrame= indexStringColumns2(data, Array("appOrSite", "interests","media", "publisher", "size", "type", "user"))

    // We apply the model to our indexed data
    val predictions: DataFrame = model.transform(userIndexer)

    // We create a dataframe with the prediction to be able later to add this column into another dataframes
    var predictionsDf: DataFrame = predictions.select("prediction")

    // We add ids to each row of these dataframes to be able to merge them together
    val newDf: DataFrame = initDf.withColumn("id1", monotonically_increasing_id())
    val newPredictions: DataFrame = predictionsDf.withColumn("id2", monotonically_increasing_id())

    // Join the original dataframe with the prediction
    val df2: DataFrame = newDf.as("df1").join(newPredictions.as("df2")
      , newDf("id1") === newPredictions("id2"),
      "inner")
        .select("df1.appOrSite", "df1.bidFloor", "df1.city", "df1.exchange"
          , "df1.impid", "df1.interests", "df1.media", "df1.network", "df1.os"
          , "df1.publisher", "df1.size", "df1.timestamp", "df1.type", "df1.user", "df2.prediction")

    // Transform the column size into a column of string to be able to write the dataframe into a csv file.
    val dfToExport: DataFrame = DataCleaner.castSize(df2)

    // Export the result as CSV in one file
    dfToExport.coalesce(1).write.format("csv").option("header", "true").save("./predictions/prediction.csv")
  }

  // ------------------------- FUNCTIONS --------------------------

    /**
    * creates a new dataFrame from the current dataFrame to index each columns specified in cols
    * 
    * @param df dataFrame to use
    * @param cols columns that we want to index
    */
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
    /**
    * takes a DataFrame and add a new column that contains the labels's weight
    * 
    * @param dataset dataFrame to change
    */
  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Count the number of labels that equal 0
    val numNegatives = dataset.filter(dataset("label") === 0).count

    // Count the number of lines of the dataFrame
    val datasetSize = dataset.count

    // Calculates the ratio between the numNegatives and the datasetSize
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    // Calculates the weight of each labels
    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    // Creates the new classWeightCol with the weight of each label
    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("label")))
    weightedDataset
  }

}
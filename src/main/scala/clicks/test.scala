import clicks.ClickPrediction.{assembler, testData, trainData, _}
import clicks.DataCleaner
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
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

    val df = spark.read.json("/home/godefroi/Téléchargements/data-students.json")

    val data = DataCleaner.newDf(df)

    println("-----------------------------         DATA CLEAN          ------------------------------------------------------")
    data.printSchema()
    data.show(20)


    def indexStringColumns(df : DataFrame, col: String) : DataFrame = {
        var newdf : DataFrame = df
        val si : StringIndexer = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")
        val sm: StringIndexerModel = si.fit(newdf)
        val indexed : DataFrame = sm.transform(newdf).drop(col)
        newdf = indexed

        return newdf
    }

    val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("appOrSiteIndex")
    val interestsIndexer = new StringIndexer().setInputCol("interests").setOutputCol("interestsIndex")
    val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("mediaIndex")
    val publisherIndexer = new StringIndexer().setInputCol("publisher").setOutputCol("publisherIndex")
    val userIndexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex")

   /* val appOrSiteIndexer: DataFrame = indexStringColumns(data, "appOrSite")
    val interestsIndexer: DataFrame = indexStringColumns(appOrSiteIndexer, "interests")
    val mediaIndexer: DataFrame = indexStringColumns(interestsIndexer, "media")
    val publisherIndexer: DataFrame = indexStringColumns(mediaIndexer, "publisher")
    val userIndexer: DataFrame = indexStringColumns(publisherIndexer, "user")
*/
    /*println("-----------------------------         USER INDEXER               ------------------------------------------------------")
    userIndexer.printSchema()
    userIndexer.show()
*/

    val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(Array("bidFloor", "appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "userIndex"))
        .setOutputCol("features")

    val testing = 0.2
    val training = 0.8
    val splits : Array[DataFrame] = data.randomSplit(Array(training, testing))


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


    spark.close()
}
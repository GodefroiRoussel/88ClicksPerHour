import clicks.DataCleaner
import org.apache.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

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

    //val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex")
    val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("appOrSiteIndex")
    val interestsIndexer = new StringIndexer().setInputCol("interests").setOutputCol("interestsIndex")
    val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("mediaIndex")
    val publisherIndexer = new StringIndexer().setInputCol("publisher").setOutputCol("publisherIndex")

    val appEncoder = new OneHotEncoder().setInputCol("appOrSiteIndex").setOutputCol("appOrSiteIndexEnc")
    val interestsEncoder = new OneHotEncoder().setInputCol("interestsIndex").setOutputCol("interestsIndexEnc")
    val mediaEncoder = new OneHotEncoder().setInputCol("mediaIndex").setOutputCol("mediaIndexEnc")
    val publisherEncoder = new OneHotEncoder().setInputCol("publisherIndex").setOutputCol("publisherIndexEnc")

    //val userIndexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex")

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
        .setInputCols(Array("bidFloor", "appOrSiteIndexEnc", "interestsIndexEnc", "mediaIndexEnc", "publisherIndexEnc"))
        .setOutputCol("features")

    val testing = 0.2
    val training = 0.8
    val splits : Array[DataFrame] = data.randomSplit(Array(training, testing))


    val trainData : DataFrame = splits(0)
    val testData : DataFrame = splits(1)

    trainData.printSchema()
    trainData.show()

    testData.printSchema()
    testData.show()


    //var train: DataFrame = assembler.transform(trainData)


    //var test: DataFrame = assembler.transform(testData)


    val lr: LogisticRegression = new LogisticRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(10)
        .setRegParam(0.3)

    println("Je passe bien après la lr")

    val stages = Array(appOrSiteIndexer, interestsIndexer, mediaIndexer, publisherIndexer, appEncoder, interestsEncoder, mediaEncoder, publisherEncoder, assembler, lr)

    val pipeline = new Pipeline().setStages(stages)

    println("On build le Pipeline ça va tout crash !!")

   val model: PipelineModel = pipeline.fit(trainData)

    println("Ah ben non au final ")


   //val model: LogisticRegressionModel = lr.fit(trainData)
   //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


    // Test the model
    val predictions: DataFrame = model.transform(testData)
    predictions.printSchema()
    //predictions.show()

    println("C'était ce putain de SHOW ?")
    //predictions.select ("label", "prediction","rawPrediction").show()
    //predictions.select("prediction").distinct().show()
    //predictions.select("label").distinct().show()
    val evaluator = new BinaryClassificationEvaluator()
        .setMetricName("areaUnderROC")
        .setRawPredictionCol("rawPrediction")
        .setLabelCol("label")

    val eval = evaluator.evaluate(predictions)
    println("Test set areaunderROC/accuracy = " + eval)


    spark.close()
}
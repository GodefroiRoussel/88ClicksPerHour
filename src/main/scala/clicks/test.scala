import clicks.ClickPrediction._
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

    val bidFloorIndexer: DataFrame = indexStringColumns(data, "bidFloor")
    val appOrSiteIndexer: DataFrame = indexStringColumns(bidFloorIndexer, "appOrSite")
    val interestsIndexer: DataFrame = indexStringColumns(appOrSiteIndexer, "interests")
    val mediaIndexer: DataFrame = indexStringColumns(interestsIndexer, "media")
    val publisherIndexer: DataFrame = indexStringColumns(mediaIndexer, "publisher")
    //val sizeIndexer: DataFrame = indexStringColumns(publisherIndexer, "size")
    //val timestampIndexer: DataFrame = indexStringColumns(sizeIndexer, "timestamp")
    val userIndexer: DataFrame = indexStringColumns(publisherIndexer, "user")
    println("-----------------------------         USER INDEXER               ------------------------------------------------------")
    userIndexer.printSchema()
    userIndexer.show()

    /*
        val encoder : DataFrame = oneHot(userIndexer, Array("bidFloorIndex", "appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex"))
        println(" -----------------------------          ENCODER               ------------------------------------------------------")
        encoder.printSchema()
        encoder.show()*/

    val test = 1
    val training = 1
    val splits : Array[DataFrame] = userIndexer.randomSplit(Array(training, test))
    val trainData : DataFrame = splits(0)
    val testData : DataFrame = splits(1)

    val assembler: VectorAssembler = new VectorAssembler()
        .setInputCols(Array("bidFloorIndex", "appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "userIndex"))
        .setOutputCol("features")

    val OneHotTRAIN: DataFrame = assembler.transform(trainData)
    OneHotTRAIN.printSchema()

    val OneHotTEST: DataFrame = assembler.transform(testData)
    OneHotTEST.printSchema()
    // Train the model

    val lr: LogisticRegression = new LogisticRegression()
        .setLabelCol("label")
         .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(10)
        .setRegParam(0.3)
        .setThreshold(0.5)
        .setFamily("auto")

    val stages = Array(assembler, lr)

    val pipeline = new Pipeline().setStages(stages)

   val model = pipeline.fit(trainData)

   //val model: LogisticRegressionModel = lr.fit(trainData)
   //println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")


    // Test the model
    val predictions: DataFrame = model.transform(testData)
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
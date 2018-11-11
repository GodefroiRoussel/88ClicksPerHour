import org.apache.spark
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

object ClickPrediction extends App {
    val spark = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()

    val data = spark.read.json("/home/godefroi/Téléchargements/new2JSON.json")


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
    val timestampIndexer: DataFrame = indexStringColumns(sizeIndexer, "timestamp")
    val userIndexer: DataFrame = indexStringColumns(timestampIndexer, "user")
    userIndexer.printSchema


    val encoder : DataFrame = oneHot(userIndexer, Array("appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex"))
    println("ENCODER")
    encoder.printSchema

    val test = 0.01;
    val training = 0.99;
    val splits : Array[DataFrame] = encoder.randomSplit(Array(training, test))
	  val trainData : DataFrame = splits(0)
	  val testData : DataFrame = splits(1)

    println("TRAIN DATA")
    trainData.printSchema()
    println("TEST DATA SUMMARY "+testData.count)


    val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("bidFloor", "appOrSiteIndexEnc", "interestsIndexEnc", "mediaIndexEnc", "publisherIndexEnc", "sizeIndexEnc", "timestampIndexEnc", "userIndexEnc"))
        .setOutputCol("features")

    val OneHotTRAIN: DataFrame = assembler.transform(trainData)
    val OneHotTEST: DataFrame = assembler.transform(testData)

        // Train the model

    val lr: LogisticRegression = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(1).setRegParam(0.3).setElasticNetParam(0.8)
    val model: LogisticRegressionModel = lr.fit(OneHotTEST)
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    /*
    // Test the model
    val predictions = model.transform(OneHotTEST)
    predictions.show*/

    spark.close()
}

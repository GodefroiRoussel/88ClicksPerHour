import org.apache.spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object ClickPrediction extends App {
    val spark = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()

    val data = spark.read.json("/home/godefroi/Téléchargements/new2JSON.json")


    def indexStringColumns(df : DataFrame, col: String) : DataFrame = {
        var newdf = df
        val si = new StringIndexer().setInputCol(col).setOutputCol(col + "Index")
        val sm: StringIndexerModel = si.fit(newdf)
        val indexed = sm.transform(newdf).drop(col)
        newdf = indexed

        return newdf
    }



    // One hot encoder estimator
    // Maps a column of label indices to a column of binary vectors, with at most a single one-value. That alows Logistic Regression to use categorical features
    def oneHot(df : DataFrame, cols: Array[String]) : DataFrame ={

        var newdf = df

        for (col <- cols){

            val oh = new OneHotEncoderEstimator()
                .setInputCols(Array(col))
                .setOutputCols(Array(col + "Enc"))
            val model = oh.fit(newdf)
            val encoded = model.transform(newdf).drop(col)
            newdf = encoded

        }
        return newdf

    }

    val appOrSiteIndexer = indexStringColumns(data, "appOrSite")
    val interestsIndexer = indexStringColumns(appOrSiteIndexer, "interests")
    val mediaIndexer = indexStringColumns(interestsIndexer, "media")
    val publisherIndexer = indexStringColumns(mediaIndexer, "publisher")
    val sizeIndexer = indexStringColumns(publisherIndexer, "size")
    val timestampIndexer = indexStringColumns(sizeIndexer, "timestamp")
    val userIndexer = indexStringColumns(timestampIndexer, "user")
    userIndexer.printSchema

    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("label")
    val sm: StringIndexerModel = labelIndexer.fit(userIndexer)
    val labelIndexed = sm.transform(userIndexer)

    val encoder =oneHot(labelIndexed, Array("appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex"))
    encoder.printSchema

    val test = 0.25;
    val training = 0.75;
    val splits = encoder.randomSplit(Array(training, test))
	  val trainData = splits(0)
	  val testData = splits(1)

    val assembler = new VectorAssembler().setInputCols(Array("bidFloor", "appOrSiteIndexEnc", "interestsIndexEnc", "mediaIndexEnc", "publisherIndexEnc", "sizeIndexEnc", "timestampIndexEnc", "userIndexEnc"))
        .setOutputCol("features")

    val OneHotTRAIN = assembler.transform(trainData)
    val OneHotTEST = assembler.transform(testData)

        // Train the model
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(OneHotTRAIN)   
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Test the model
    val predictions = model.transform(OneHotTEST)
    predictions.show

    spark.close()
}

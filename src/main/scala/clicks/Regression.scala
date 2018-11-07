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

    val encoder =oneHot(userIndexer , Array("appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex"))
    encoder.printSchema

    val assembler = new VectorAssembler().setInputCols(Array("bidFloor", "appOrSiteIndexEnc", "interestsIndexEnc", "mediaIndexEnc", "publisherIndexEnc", "sizeIndexEnc", "timestampIndexEnc", "userIndexEnc"))
        .setOutputCol("features")

    val data3 = assembler.transform(encoder)
    data3.printSchema()
    data3.show(20)
    data3.select("features").show(20, false)

    spark.close()
}

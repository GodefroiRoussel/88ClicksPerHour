<<<<<<< HEAD
import clicks.DataCleaner
import org.apache.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object test2 extends App {
    val spark = SparkSession
        .builder()
        .appName("Click prediction")
        .config("spark.master", "local")
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.json("/Users/assilelyahyaoui/Documents/data-students.json")
    val data = DataCleaner.newDf(df)
    val categoricalColumns = Array("appOrSite", "interests", "media", "publisher")
    val stringIndexers = categoricalColumns.map {colName =>
    new StringIndexer().setInputCol(colName).setOutputCol(colName + "Index").fit(data)}
    val encoders = categoricalColumns.map { colName =>
    new OneHotEncoder().setInputCol(colName+"Index").setOutputCol(colName + "Enc")}

    val labeler = new Bucketizer().setInputCol("label")
    .setOutputCol("label")
    .setSplits(Array(0.0, 40.0, Double.PositiveInfinity))

    val assembler = new VectorAssembler().setInputCols("bidFloor", "appOrSiteEnc", "interestsEnc", "mediaEnc", "publisherEnc")
    .setOutputCol("features")

    val dTree = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxBins(7000)

    val stages = stringIndexers ++ encoders ++ Array(labeler, assembler, dTree)
    val pipeline = new Pipeline().setStages(stages)

    val paramGrid = new ParamGridBuilder.addGrid(dTree.maxDepth, Array(4,5,6)).build()

    val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

    val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator)
    .setEstimatorParamsMaps(paramGrid).setNumFolds(3)

    val testValue = 0.25
    val training = 0.75
    val splits : Array[DataFrame] = data.randomSplit(Array(training, testValue))
    var trainData : DataFrame = splits(0)
    var testData : DataFrame = splits(1)

    val model = crossval.fit(trainData)
    val predictions = model.transform(testData)
    predictions.printSchema()
    predictions.select("prediction").distinct().show()
    predictions.select("label").distinct().show()

    spark.close()
}
=======
>>>>>>> 5423a894e3e2ad5a0126e4248a53f495eef25829

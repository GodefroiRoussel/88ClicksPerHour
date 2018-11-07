def trainModel()  = {
	
	import org.apache.spark.ml._
	import org.apache.spark.ml.attribute._
	import org.apache.spark.ml.feature._
	import org.apache.spark.ml.classification._
	import org.apache.spark.ml.regression.GBTRegressor

    def indexStringColumns(df : DataFrame, col: String) : DataFrame ={
      var newdf = df
      val si = new StringIndexer().setInputCol(col).setOutputCol(col +"Index")
      val sm :  StringIndexerModel = si.fit(newdf)
      val indexed =  sm.transform(newdf).drop(col)
        newdf = indexed
      }
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
	val data = spark.read.json("Users/Marion/Desktop/data-students-new2/data.json")

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

	
/**
	val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("appOrSiteIndex")
	val interestsIndexer = new StringIndexer().setInputCol("interests").setOutputCol("interestsIndex")
	val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("mediaIndex")
	val publisherIndexer = new StringIndexer().setInputCol("publisher").setOutputCol("publisherIndex")
	val sizeIndexer = new StringIndexer().setInputCol("size").setOutputCol("sizeIndex")
	val timestampIndexer = new StringIndexer().setInputCol("timestamp").setOutputCol("timestampIndex")
	val userIndexer = new StringIndexer().setInputCol("user").setOutputCol("userIndex")*/

	/*val encoder = new OneHotEncoderEstimator().setInputCols(Array("appOrSiteIndex", "interestsIndex", "mediaIndex", "publisherIndex", "sizeIndex", "timestampIndex", "userIndex"))
	.setOutputCols(Array("appOrSiteEnc", "interestsEnc", "mediaEnc", "publisherEnc", "sizeEnc", "timestampEnc", "userEnc"))*/

	val assembler = new VectorAssembler().setInputCols(Array("bidFloor", "appOrSiteEnc", "interestsEnc", "mediaEnc", "publisherEnc", "sizeEnc", "timestampEnc", "userEnc"))
	.setOutputCol("features")

    val df2 = assembler.transform(data)
    df2.show
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    df3.show
    val Array(trainingData, testData) = df3.randomSplit(Array(0.8, 0.2))

    // Train the model
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(trainingData)   
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Test the model
    val predictions = model.transform(testData)
    predictions.show
}

// ANCIEN
/**
	val labelColumn = "label"
	val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

	val gbt = new GBTRegressor()
	    .setLabelCol(labelColumn)
	    .setFeaturesCol("features")
	    .setPredictionCol("Predicted " + labelColumn)
	    .setMaxIter(50)

	// The stages of our ML pipeline 
	val stages = Array(stringIndexers, assembler, gbt)
	val pipeline = new Pipeline().setStages(stages)
	val model = pipeline.fit(trainingData)
	val predictions = model.transform(testData)
	predictions.select("features", "label").show()

    /** val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(trainingData) 
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}") **/





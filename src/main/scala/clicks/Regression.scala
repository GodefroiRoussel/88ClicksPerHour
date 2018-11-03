import org.apache.spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler

// String indexer
// Encodes a string column of labels to a column of label indices. The most frequent label gets index 0

def indexStringColumns(df : DataFrame, cols: Array[String]) : DataFrame ={
  var newdf = df

  for (col <- cols){
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
      .setInputCols(col)
      .setOutputCols(col + "Vec")
    val model = oh.fit(newdf)
    val encoded = model.transform(newdf).drop(col)
    newdf = encoded

  }
  return newdf

}

// Vector assembler 
// Combines a given list of columns into a single vector column.
/**
  def vectorAssembler(df : DataFrame, cols: Array[String]) : DataFrame ={

  var newdf = df

  val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCols("features")

  val output = assembler.transform(newdf)
  newdf = output

  }
  return newdf

}
**/



val data = spark.read.json("Users/Marion/Desktop/data-students-new2/data.json")
data.show(50, false)

val df = indexStringColumns(data , Array("appOrSite", "media", "publisher", "size" ,"timestamp", "user", "interests"))

df.printSchema
val df2 =oneHot(df , Array("appOrSiteIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex", "interestsIndex"))
println("aalal")
df2.printSchema
df2.show(50, false )

/**
val df3 = vectorAssembler(df2, Array("appOrSiteIndexVec", "mediaIndexVec", "publisherIndexVec", "sizeIndexVec" ,"timestampIndexVec", "userIndexVec", "interestsIndexVec", "bidFloor"))
df3.printSchema
*//

/**
// Load training data
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)**/

// Fit the model
//val lrModel = lr.fit(data)

// Print the coefficients and intercept for logistic regression
//println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


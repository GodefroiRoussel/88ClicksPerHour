import org.apache.spark
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel



def indexStringColumns(df : DataFrame, cols: Array[String]) : DataFrame ={
  var newdf = df

  for (col <- cols){
    val si = new StringIndexer().setInputCol(col).setOutputCol(col +"Index")

    val sm :  StringIndexerModel = si.fit(newdf)
    val indexed =  sm.transform(newdf).drop(col)
    newdf = indexed


    //newdf = newdf.withColumnRenamed(col + "Index" , col)
  }
  return newdf
}

def oneHot(df : DataFrame, cols: Array[String]) : DataFrame ={

  var newdf = df

  for (col <- cols){

    val oh = new OneHotEncoderEstimator()
      .setInputCols(Array(col))
      .setOutputCols(Array(col +"Vec"))
    val model = oh.fit(newdf)
    val encoded = model.transform(newdf).drop(col)
    newdf = encoded
    //newdf = newdf.withColumnRenamed(col + "Vec", col)

  }
  return newdf

}



val data = spark.read.json("./json/data-students-new.json")
data.show(50, false)

val df = indexStringColumns(data , Array("appOrSite", "media", "publisher", "size" ,"timestamp", "user", "interests"))

df.printSchema
val df2 =oneHot(df , Array("appOrSiteIndex", "mediaIndex", "publisherIndex", "sizeIndex" ,"timestampIndex", "userIndex", "interestsIndex"))
println("aalal")
df2.printSchema
df2.show(50, false )


// Vector assembleur

val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)







// Load training data



//val lr = new LogisticRegression()
//.setMaxIter(10)
//.setRegParam(0.3)
//.setElasticNetParam(0.8)

// Fit the model
//val lrModel = lr.fit(data)

// Print the coefficients and intercept for logistic regression
//println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


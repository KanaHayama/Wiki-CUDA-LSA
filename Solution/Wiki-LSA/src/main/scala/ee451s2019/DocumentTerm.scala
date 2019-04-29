package ee451s2019

import org.apache.spark.sql.{Dataset, SparkSession}

object DocumentTerm {

	def get(sparkSession: SparkSession, documentContentMatrix: Dataset[(String, String)], stopWords: Set[String] ) : Dataset[(String, Seq[String])] = {
		import sparkSession.implicits._
		val broadcastedStopWords = sparkSession.sparkContext.broadcast(stopWords)

		documentContentMatrix.mapPartitions { keyValues =>
			val separator = new LemmaSeparator(broadcastedStopWords.value) // partition owned instance
			keyValues.map { case (title, content) =>
				(title, separator.get(content))
			}
		}
	}
}
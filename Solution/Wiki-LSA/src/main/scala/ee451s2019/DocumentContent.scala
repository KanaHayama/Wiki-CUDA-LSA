package ee451s2019

import edu.umd.cloud9.collection.XMLInputFormat
import edu.umd.cloud9.collection.wikipedia.WikipediaPage
import edu.umd.cloud9.collection.wikipedia.language.EnglishWikipediaPage
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.sql.{Dataset, SparkSession}

object DocumentContent {

	private def isGoodArticle(page: WikipediaPage) : Boolean = {
		!page.isEmpty && page.isArticle && !page.isRedirect && !page.isDisambiguation && !page.getTitle.contains("(disambiguation)")
	}

	private def parseArticleXML(xml: String): Option[(String, String)] = { // TODO: make my own version of WikipediaPage & EnglishWikipediaPage base on its source code
		val page = new EnglishWikipediaPage()
		val hackedPageXml = xml.replaceFirst("<text xml:space=\"preserve\" bytes=\"\\d+\">", "<text xml:space=\"preserve\">")
		WikipediaPage.readPage(page, hackedPageXml)
		if (isGoodArticle(page)) {
			Some((page.getTitle, page.getContent))
		} else {
			None
		}
	}

	def get(sparkSession: SparkSession, xmlFilename: String) : Dataset[(String, String)] = { // why Dataset? // TODO: repalce XMLInputFormat
		import sparkSession.implicits._
		val hadoopConfig = new Configuration()
		hadoopConfig.set(XMLInputFormat.START_TAG_KEY, "<page>")
		hadoopConfig.set(XMLInputFormat.END_TAG_KEY, "</page>")
		val wikiFile = sparkSession.sparkContext.newAPIHadoopFile(xmlFilename, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], hadoopConfig)
		val wikiXml = wikiFile.map(_._2.toString).toDS()

		wikiXml.filter(_ != null).flatMap(parseArticleXML)
	}
}

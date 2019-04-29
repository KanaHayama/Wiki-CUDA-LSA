package ee451s2019

import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}

import scala.collection.mutable.ArrayBuffer

class LemmaSeparator(private val stopWords: Set[String]) {

	private final val ANNOTATORS = "tokenize, ssplit, pos, lemma"

	private val properties = new Properties()
	properties.put("annotators", ANNOTATORS)
	private val nlpPipeline = new StanfordCoreNLP(properties)

	private def isGoodLemma(lemma: String) : Boolean = {
		lemma.length > 2 && !stopWords.contains(lemma) && lemma.forall(c => Character.isLetter(c))
	}

	def get(content: String) : Seq[String] = {
		import scala.collection.JavaConverters._
		val annotation = new Annotation(content)
		nlpPipeline.annotate(annotation)
		val result = new ArrayBuffer[String]()
		for (sentence <- annotation.get(classOf[SentencesAnnotation]).asScala) {
			for (token <- sentence.get(classOf[TokensAnnotation]).asScala) {
				val lemma = token.get(classOf[LemmaAnnotation])
				if (isGoodLemma(lemma)) {
					result += lemma.toLowerCase
				}
			}
		}
		result
	}
}
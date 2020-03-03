// Jar file required from Advanced Analytics with Spark: ch06-lsa
// startup command:
// spark-shell --jars wiki/target/ch06-lsa-2.0.0-jar-with-dependencies.jar --master local[2]


import edu.umd.cloud9.collection.XMLInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._
val path = "wiki/wikidump.xml"
@transient val conf = new Configuration()
conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
val kvs = spark.
  sparkContext.
  newAPIHadoopFile(
    path,
    classOf[XMLInputFormat],
    classOf[LongWritable],
    classOf[Text],
    conf)
val rawXmls = kvs.
  map(_._2.toString).
  toDS()

import edu.umd.cloud9.collection.wikipedia.language._
import edu.umd.cloud9.collection.wikipedia._
def wikiXmlToPlainText(pageXml: String): Option[(String, String)] = {
  // Wikipedia has updated their dumps slightly since Cloud9 was written,
  // so this hacky replacement is sometimes required to get parsing to work.
  val hackedPageXml = pageXml.replaceFirst(
    "<text bytes=\"\\d+\" xml:space=\"preserve\">",
    "<text xml:space=\"preserve\">")
  val page = new EnglishWikipediaPage()
  WikipediaPage.readPage(page, hackedPageXml)
  if (page.isEmpty) None
  else Some((page.getTitle, page.getContent))
}
val docTexts = rawXmls.filter(_ != null).flatMap(wikiXmlToPlainText)

// lemmanize and filter stop words
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._
import java.util.Properties
import org.apache.spark.sql.Dataset

def createNLPPipeline(): StanfordCoreNLP = {
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit, pos, lemma")
  new StanfordCoreNLP(props)
}
def isOnlyLetters(str: String): Boolean = {
  str.forall(c => Character.isLetter(c))
}
def plainTextToLemmas(text: String, stopWords: Set[String],
  pipeline: StanfordCoreNLP): Seq[String] = {
  val doc = new Annotation(text)
  pipeline.annotate(doc)
  val lemmas = new ArrayBuffer[String]()
  val sentences = doc.get(classOf[SentencesAnnotation])
  for (sentence <- sentences.asScala;
    token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma)
      && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
  }
  lemmas
}
val stopWords = scala.io.Source.fromFile("wiki/stopwords.txt").getLines().toSet
val bStopWords = spark.sparkContext.broadcast(stopWords)
val terms: Dataset[(String, Seq[String])] =
  docTexts.mapPartitions { iter =>
    val pipeline = createNLPPipeline()
    iter.map { case(title, contents) =>
      (title, plainTextToLemmas(contents, bStopWords.value, pipeline))
    }
}

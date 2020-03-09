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

// Covert temrs to DF for spark.ml tf/idf functions
val termsDF = terms.toDF("title", "terms")
val filtered = termsDF.where(size($"terms") > 1)

// Create dataframe with sparse vectors for term frequencies
import org.apache.spark.ml.feature.CountVectorizer
val numTerms = 20000
val countVectorizer = new CountVectorizer().
  setInputCol("terms").setOutputCol("termFreqs").
  setVocabSize(numTerms)
val vocabModel = countVectorizer.fit(filtered)
val docTermFreqs = vocabModel.transform(filtered)

// Cache to save computation time
docTermFreqs.cache()

// use spark.ml to calculate IDF
import org.apache.spark.ml.feature.IDF
val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
val idfModel = idf.fit(docTermFreqs)
val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")

// Save term ids that map to terms
val termIds: Array[String] = vocabModel.vocabulary

// Make document title IDS by calling zip and relying on the fact that
// the ordering will not change.
val docIds = docTermFreqs.rdd.map(_.getString(0)).
  zipWithUniqueId().
  map(_.swap).
  collect().toMap

// Perform some vector conversions from spark.ml to spark.mlib in order to
// run SVD.

import org.apache.spark.mllib.linalg.{Vectors,
  Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
val vecRdd = docTermMatrix.select("tfidfVec").rdd.map { row =>
  Vectors.fromML(row.getAs[MLVector]("tfidfVec"))
}

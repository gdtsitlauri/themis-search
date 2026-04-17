package themis

final case class Document(id: String, title: String, text: String)
final case class Posting(termFrequency: Int, positions: Seq[Int], tfidf: Double)

final class InvertedIndex(documents: Seq[Document]) {
  private val tokenized: Map[String, Seq[String]] =
    documents.map(doc => doc.id -> tokenize(s"${doc.title} ${doc.text}")).toMap

  private val documentFrequencies: Map[String, Int] =
    tokenized.values.flatMap(_.distinct).toSeq.groupBy(identity).view.mapValues(_.size).toMap

  val postings: Map[String, Map[String, Posting]] =
    tokenized.toSeq.flatMap { case (docId, terms) =>
      terms.zipWithIndex.groupBy(_._1).map { case (term, pairs) =>
        val tf = pairs.size
        val idf = math.log((documents.size.toDouble + 1.0) / (documentFrequencies.getOrElse(term, 0) + 1.0)) + 1.0
        term -> Map(docId -> Posting(tf, pairs.map(_._2), tf * idf))
      }
    }.groupMapReduce(_._1)(_._2)(_ ++ _)

  def termLookup(term: String): Map[String, Posting] = postings.getOrElse(term.toLowerCase, Map.empty)

  def booleanAnd(terms: Seq[String]): Seq[String] =
    if (terms.isEmpty) Seq.empty
    else terms.map(termLookup(_).keySet).reduce(_ intersect _).toSeq.sorted

  def booleanOr(terms: Seq[String]): Seq[String] =
    terms.flatMap(termLookup(_).keySet).distinct.sorted

  def booleanNot(term: String): Seq[String] =
    documents.map(_.id).filterNot(termLookup(term).keySet).sorted

  def phraseQuery(phrase: String): Seq[String] = {
    val terms = tokenize(phrase)
    if (terms.isEmpty) Seq.empty
    else {
      val candidateDocs = terms.map(term => termLookup(term).keySet).reduce(_ intersect _)
      candidateDocs.toSeq.sorted.filter { docId =>
        termLookup(terms.head).getOrElse(docId, Posting(0, Seq.empty, 0.0)).positions.exists { start =>
          terms.zipWithIndex.forall { case (term, offset) =>
            termLookup(term).getOrElse(docId, Posting(0, Seq.empty, 0.0)).positions.contains(start + offset)
          }
        }
      }
    }
  }

  def nearQuery(left: String, right: String, distance: Int): Seq[String] = {
    val leftDocs = termLookup(left).keySet
    val rightDocs = termLookup(right).keySet
    (leftDocs intersect rightDocs).toSeq.sorted.filter { docId =>
      val leftPositions = termLookup(left).getOrElse(docId, Posting(0, Seq.empty, 0.0)).positions
      val rightPositions = termLookup(right).getOrElse(docId, Posting(0, Seq.empty, 0.0)).positions
      leftPositions.exists(l => rightPositions.exists(r => math.abs(l - r) <= distance))
    }
  }

  def search(query: QueryExpr): Seq[String] = query match {
    case QueryExpr.Term(value) => termLookup(value).keySet.toSeq.sorted
    case QueryExpr.And(left, right) => search(left).intersect(search(right)).sorted
    case QueryExpr.Or(left, right) => (search(left) ++ search(right)).distinct.sorted
    case QueryExpr.Not(inner) => documents.map(_.id).filterNot(search(inner).toSet).sorted
    case QueryExpr.Phrase(terms) => phraseQuery(terms.mkString(" "))
    case QueryExpr.Near(left, right, distance) => nearQuery(left, right, distance)
  }

  def exportMetadata: Map[String, Int] =
    Map("documents" -> documents.size, "terms" -> postings.keySet.size)

  def exportRows: Seq[Map[String, String]] =
    postings.toSeq.flatMap { case (term, docsForTerm) =>
      docsForTerm.toSeq.map { case (docId, posting) =>
        Map(
          "term" -> term,
          "doc_id" -> docId,
          "tf" -> posting.termFrequency.toString,
          "tfidf" -> posting.tfidf.formatted("%.4f")
        )
      }
    }

  private def tokenize(text: String): Seq[String] =
    text.toLowerCase.split("[^a-z0-9]+").filter(_.nonEmpty).toSeq
}

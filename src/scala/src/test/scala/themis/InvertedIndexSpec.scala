package themis

import org.scalatest.funsuite.AnyFunSuite
import themis.QueryExpr._

final class InvertedIndexSpec extends AnyFunSuite {
  private val docs = Seq(
    Document("d1", "Causal Search", "causal relevance search engine"),
    Document("d2", "Neural Retrieval", "hybrid semantic retrieval"),
    Document("d3", "Counterfactual Search", "counterfactual ranking causal retrieval")
  )
  private val index = new InvertedIndex(docs)

  test("term lookup returns postings") {
    assert(index.termLookup("causal").contains("d1"))
    assert(index.termLookup("causal")("d1").tfidf > 0.0)
  }

  test("boolean and intersects documents") {
    assert(index.booleanAnd(Seq("causal", "search")) == Seq("d1", "d3"))
  }

  test("boolean or unions documents") {
    assert(index.booleanOr(Seq("hybrid", "counterfactual")) == Seq("d2", "d3"))
  }

  test("boolean not removes matched documents") {
    assert(index.booleanNot("hybrid") == Seq("d1", "d3"))
  }

  test("phrase query finds ordered positions") {
    assert(index.phraseQuery("causal relevance").contains("d1"))
  }

  test("near query finds nearby tokens") {
    assert(index.nearQuery("counterfactual", "retrieval", 3).contains("d3"))
  }

  test("query expression search works") {
    val result = index.search(And(Term("causal"), Or(Term("search"), Term("retrieval"))))
    assert(result == Seq("d1", "d3"))
  }
}

package themis

sealed trait QueryExpr

object QueryExpr {
  final case class Term(value: String) extends QueryExpr
  final case class And(left: QueryExpr, right: QueryExpr) extends QueryExpr
  final case class Or(left: QueryExpr, right: QueryExpr) extends QueryExpr
  final case class Not(query: QueryExpr) extends QueryExpr
  final case class Phrase(terms: Seq[String]) extends QueryExpr
  final case class Near(left: String, right: String, distance: Int) extends QueryExpr
}

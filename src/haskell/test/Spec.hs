module Main where

import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as T
import Themis.Query (Query (..), evaluateQuery, optimizeQuery, optimizerPreservesSemantics, parseQueryText, selectivityEstimate, serializeElasticsearch, serializeQuery, serializeSQL)

assert :: Bool -> String -> IO ()
assert cond msg = if cond then pure () else error msg

docs :: Map.Map String [String]
docs =
  Map.fromList
    [ ("d1", ["machine", "learning", "deep", "retrieval"]),
      ("d2", ["neural", "network", "training", "retrieval"]),
      ("d3", ["symbolic", "logic", "formal", "semantics"])
    ]

main :: IO ()
main = do
  let original = And (Term "machine") (Or (Term "deep") (Term "neural"))
  let roundtrip = parseQueryText (T.unpack (serializeQuery original))
  assert (roundtrip == Right original) "parse roundtrip failed"
  let implicitAnd = parseQueryText "machine learning"
  assert (implicitAnd == Right (And (Term "machine") (Term "learning"))) "implicit AND parsing failed"
  let optimized = optimizeQuery (Not (Or (Term "a") (Term "b")))
  assert (optimized == And (Not (Term "a")) (Not (Term "b"))) "optimizer failed"
  let notPush = optimizeQuery (Not (Not (Term "x")))
  assert (notPush == Term "x") "double not simplification failed"
  let selectivity = selectivityEstimate (And (Term "machine") (Phrase ["deep", "retrieval"]))
  assert (selectivity > 0.0 && selectivity < 1.0) "selectivity estimate should be bounded"
  assert (optimizerPreservesSemantics docs (Not (Or (Term "machine") (Term "deep")))) "optimizer semantics check failed"
  let phraseMatches = evaluateQuery docs (Phrase ["neural", "network"])
  assert (phraseMatches == Set.fromList ["d2"]) "phrase semantics failed"
  let sql = serializeSQL (And (Term "machine") (Term "deep"))
  assert ("AND" `elem` words (T.unpack sql)) "sql serializer failed"
  let es = serializeElasticsearch (Term "machine")
  assert (show es /= "{}") "elasticsearch serializer failed"
  putStrLn "themis-query-test passed"

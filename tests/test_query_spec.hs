module Main where

import qualified Data.Map.Strict as Map
import qualified Data.Text as T
import Themis.Query (Query (..), optimizerPreservesSemantics, parseQueryText, selectivityEstimate, serializeQuery)

assert :: Bool -> String -> IO ()
assert cond msg = if cond then pure () else error msg

main :: IO ()
main = do
  let query = And (Term "machine") (Or (Term "deep") (Term "neural"))
  case parseQueryText (T.unpack (serializeQuery query)) of
    Left _ -> error "query parser should accept serialized query text"
    Right _ -> pure ()
  case parseQueryText "machine learning" of
    Right (And (Term "machine") (Term "learning")) -> pure ()
    _ -> error "implicit AND query should parse"
  let docs = Map.fromList [("d1", ["machine", "deep"]), ("d2", ["neural", "training"])]
  assert (optimizerPreservesSemantics docs (Not (Or (Term "machine") (Term "deep")))) "optimizer should preserve semantics"
  assert (selectivityEstimate (Phrase ["neural", "network"]) < 1.0) "phrase selectivity should be bounded"
  putStrLn "tests/test_query_spec.hs passed"

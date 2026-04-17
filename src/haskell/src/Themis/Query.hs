{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Themis.Query
  ( Query (..),
    parseQueryText,
    optimizeQuery,
    selectivityEstimate,
    serializeQuery,
    serializeSQL,
    serializeElasticsearch,
    evaluateQuery,
    optimizerPreservesSemantics,
    payloadFromQuery,
  )
where

import Control.Applicative (empty, many, optional, some, (<|>))
import Data.Aeson (ToJSON (toJSON), Value, object, (.=))
import Data.Char (toLower, toUpper)
import Data.List (intercalate, nub)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Text (Text)
import qualified Data.Text as T
import Data.Void (Void)
import GHC.Generics (Generic)
import Text.Megaparsec (Parsec, between, eof, errorBundlePretty, lookAhead, manyTill, notFollowedBy, parse, satisfy, try)
import Text.Megaparsec.Char (alphaNumChar, char, letterChar, space1, string')
import qualified Text.Megaparsec.Char.Lexer as L

data Query
  = Term String
  | And Query Query
  | Or Query Query
  | Not Query
  | Phrase [String]
  | Near String String Int
  | Boost Query Double
  deriving (Show, Eq, Generic)

type Parser = Parsec Void String
type DocumentUniverse = Map.Map String [String]

instance ToJSON Query where
  toJSON (Term value) = object ["type" .= ("Term" :: String), "value" .= value]
  toJSON (And left right) = object ["type" .= ("And" :: String), "left" .= left, "right" .= right]
  toJSON (Or left right) = object ["type" .= ("Or" :: String), "left" .= left, "right" .= right]
  toJSON (Not inner) = object ["type" .= ("Not" :: String), "query" .= inner]
  toJSON (Phrase values) = object ["type" .= ("Phrase" :: String), "terms" .= values]
  toJSON (Near left right n) = object ["type" .= ("Near" :: String), "left" .= left, "right" .= right, "distance" .= n]
  toJSON (Boost inner weight) = object ["type" .= ("Boost" :: String), "query" .= inner, "weight" .= weight]

sc :: Parser ()
sc = L.space space1 empty empty

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbolCI :: String -> Parser String
symbolCI token = lexeme (string' token)

identifier :: Parser String
identifier =
  lexeme $
    fmap (map toLower) $
      (:) <$> (letterChar <|> char '_') <*> many (alphaNumChar <|> char '_' <|> char '-')

identifierBare :: Parser String
identifierBare =
  fmap (map toLower) $
    (:) <$> (letterChar <|> char '_') <*> many (alphaNumChar <|> char '_' <|> char '-')

phraseText :: Parser String
phraseText = fmap (map toLower) (char '"' *> manyTill (satisfy (/= '"')) (char '"'))

nearOperand :: Parser String
nearOperand = try phraseText <|> identifier

termParser :: Parser Query
termParser = Term <$> identifier

phraseParser :: Parser Query
phraseParser = Phrase . words <$> try phraseText

nearParser :: Parser Query
nearParser = try $ do
  left <- nearOperand
  _ <- sc
  _ <- string' "NEAR/"
  distance <- L.decimal
  _ <- sc
  right <- nearOperand
  pure (Near left right distance)

baseParser :: Parser Query
baseParser =
  try nearParser
    <|> between (symbolCI "(") (symbolCI ")") exprParser
    <|> phraseParser
    <|> termParser

factorParser :: Parser Query
factorParser = do
  base <- baseParser
  maybeWeight <- optional (try (char '^' *> L.float))
  pure $ maybe base (Boost base) maybeWeight

notParser :: Parser Query
notParser =
  (Not <$> (symbolCI "NOT" *> factorParser))
    <|> factorParser

andParser :: Parser Query
andParser = do
  first <- notParser
  rest <- many (try (symbolCI "AND" *> notParser) <|> try implicitAndStep)
  pure (foldl And first rest)

exprParser :: Parser Query
exprParser = chainBinary andParser "OR" Or

chainBinary :: Parser Query -> String -> (Query -> Query -> Query) -> Parser Query
chainBinary parser keyword ctor = do
  first <- parser
  rest <- many (try (symbolCI keyword *> parser))
  pure (foldl ctor first rest)

implicitAndStep :: Parser Query
implicitAndStep = do
  _ <- lookAhead implicitAndStart
  notParser

implicitAndStart :: Parser String
implicitAndStart = do
  notFollowedBy (char ')')
  token <-
    lookAhead $
      try (keywordExact "AND")
        <|> try (keywordExact "OR")
        <|> try (keywordExact "NOT")
        <|> ("(" <$ char '(')
        <|> ("\"" <$ char '"')
        <|> identifierBare
  case map toUpper token of
    "AND" -> fail "explicit AND"
    "OR" -> fail "OR boundary"
    _ -> pure token

keywordExact :: String -> Parser String
keywordExact token = try $ do
  parsed <- string' token
  notFollowedBy (alphaNumChar <|> char '_' <|> char '-')
  pure parsed

parseQueryText :: String -> Either String Query
parseQueryText input =
  case parse (optional sc *> exprParser <* optional sc <* eof) "<query>" input of
    Left err -> Left (errorBundlePretty err)
    Right q -> Right (optimizeQuery q)

optimizeQuery :: Query -> Query
optimizeQuery (And left right) =
  let optimizedLeft = optimizeQuery left
      optimizedRight = optimizeQuery right
      ordered =
        if selectivityEstimate optimizedLeft <= selectivityEstimate optimizedRight
          then And optimizedLeft optimizedRight
          else And optimizedRight optimizedLeft
   in flattenAnd ordered
optimizeQuery (Or left right) =
  let optimizedLeft = optimizeQuery left
      optimizedRight = optimizeQuery right
      ordered =
        if selectivityEstimate optimizedLeft >= selectivityEstimate optimizedRight
          then Or optimizedLeft optimizedRight
          else Or optimizedRight optimizedLeft
   in flattenOr ordered
optimizeQuery (Not (Not inner)) = optimizeQuery inner
optimizeQuery (Not (And left right)) = optimizeQuery (Or (Not left) (Not right))
optimizeQuery (Not (Or left right)) = optimizeQuery (And (Not left) (Not right))
optimizeQuery (Boost inner weight)
  | weight == 1.0 = optimizeQuery inner
  | otherwise = Boost (optimizeQuery inner) weight
optimizeQuery other = other

flattenAnd :: Query -> Query
flattenAnd (And left right)
  | left == right = left
  | otherwise =
      case right of
        And inner other -> flattenAnd (And (And left inner) other)
        _ -> And left right
flattenAnd other = other

flattenOr :: Query -> Query
flattenOr (Or left right)
  | left == right = left
  | otherwise =
      case right of
        Or inner other -> flattenOr (Or (Or left inner) other)
        _ -> Or left right
flattenOr other = other

selectivityEstimate :: Query -> Double
selectivityEstimate (Term _) = 0.20
selectivityEstimate (Phrase values) = max 0.02 (0.08 / fromIntegral (max 1 (length values)))
selectivityEstimate (Near _ _ distance) = max 0.02 (0.18 / fromIntegral (max 1 distance))
selectivityEstimate (Boost inner _) = selectivityEstimate inner
selectivityEstimate (Not inner) = max 0.0 (min 1.0 (1.0 - selectivityEstimate inner))
selectivityEstimate (And left right) = max 0.0 (min 1.0 (selectivityEstimate left * selectivityEstimate right))
selectivityEstimate (Or left right) =
  let l = selectivityEstimate left
      r = selectivityEstimate right
   in max 0.0 (min 1.0 (l + r - l * r))

serializeQuery :: Query -> Text
serializeQuery (Term value) = T.pack value
serializeQuery (And left right) = T.concat ["(", serializeQuery left, " AND ", serializeQuery right, ")"]
serializeQuery (Or left right) = T.concat ["(", serializeQuery left, " OR ", serializeQuery right, ")"]
serializeQuery (Not inner) = T.concat ["NOT ", serializeQuery inner]
serializeQuery (Phrase values) = T.pack ("\"" <> intercalate " " values <> "\"")
serializeQuery (Near left right n) = T.pack (left <> " NEAR/" <> show n <> " " <> right)
serializeQuery (Boost inner weight) = T.concat [serializeQuery inner, "^", T.pack (show weight)]

serializeSQL :: Query -> Text
serializeSQL (Term value) = T.pack ("term = '" <> value <> "'")
serializeSQL (And left right) = T.concat ["(", serializeSQL left, " AND ", serializeSQL right, ")"]
serializeSQL (Or left right) = T.concat ["(", serializeSQL left, " OR ", serializeSQL right, ")"]
serializeSQL (Not inner) = T.concat ["NOT (", serializeSQL inner, ")"]
serializeSQL (Phrase values) = T.pack ("phrase = '" <> intercalate " " values <> "'")
serializeSQL (Near left right n) = T.pack ("near('" <> left <> "', '" <> right <> "', " <> show n <> ")")
serializeSQL (Boost inner weight) = T.concat ["BOOST(", serializeSQL inner, ", ", T.pack (show weight), ")"]

serializeElasticsearch :: Query -> Value
serializeElasticsearch (Term value) = object ["term" .= object ["text" .= value]]
serializeElasticsearch (And left right) = object ["bool" .= object ["must" .= [serializeElasticsearch left, serializeElasticsearch right]]]
serializeElasticsearch (Or left right) = object ["bool" .= object ["should" .= [serializeElasticsearch left, serializeElasticsearch right]]]
serializeElasticsearch (Not inner) = object ["bool" .= object ["must_not" .= [serializeElasticsearch inner]]]
serializeElasticsearch (Phrase values) = object ["match_phrase" .= object ["text" .= intercalate " " values]]
serializeElasticsearch (Near left right n) =
  object
    [ "span_near" .=
        object
          [ "clauses" .=
              [ object ["span_term" .= object ["text" .= left]],
                object ["span_term" .= object ["text" .= right]]
              ],
            "slop" .= n
          ]
    ]
serializeElasticsearch (Boost inner weight) =
  object ["function_score" .= object ["query" .= serializeElasticsearch inner, "boost" .= weight]]

evaluateQuery :: DocumentUniverse -> Query -> Set.Set String
evaluateQuery universe (Term value) =
  Set.fromList [docId | (docId, terms) <- Map.toList universe, value `elem` terms]
evaluateQuery universe (And left right) =
  evaluateQuery universe left `Set.intersection` evaluateQuery universe right
evaluateQuery universe (Or left right) =
  evaluateQuery universe left `Set.union` evaluateQuery universe right
evaluateQuery universe (Not inner) =
  Set.fromList (Map.keys universe) `Set.difference` evaluateQuery universe inner
evaluateQuery universe (Phrase values) =
  Set.fromList [docId | (docId, terms) <- Map.toList universe, containsPhrase values terms]
evaluateQuery universe (Near left right distance) =
  Set.fromList [docId | (docId, terms) <- Map.toList universe, isNear left right distance terms]
evaluateQuery universe (Boost inner _) = evaluateQuery universe inner

containsPhrase :: [String] -> [String] -> Bool
containsPhrase phraseTerms terms =
  any (\offset -> take (length phraseTerms) (drop offset terms) == phraseTerms) [0 .. max 0 (length terms - length phraseTerms)]

isNear :: String -> String -> Int -> [String] -> Bool
isNear left right distance terms =
  or
    [ abs (l - r) <= distance
      | l <- positions left,
        r <- positions right
    ]
  where
    positions token = [idx | (idx, value) <- zip [0 ..] terms, value == token]

optimizerPreservesSemantics :: DocumentUniverse -> Query -> Bool
optimizerPreservesSemantics universe query =
  evaluateQuery universe query == evaluateQuery universe (optimizeQuery query)

payloadFromQuery :: String -> Maybe Query -> [String] -> Value
payloadFromQuery raw maybeQuery errors =
  object
    [ "raw" .= raw,
      "normalized" .= maybe raw (T.unpack . serializeQuery . optimizeQuery) maybeQuery,
      "ast" .= maybe (object ["type" .= ("RawQuery" :: String)]) toJSON maybeQuery,
      "estimated_selectivity" .= maybe (1.0 :: Double) selectivityEstimate maybeQuery,
      "sql" .= maybe ("" :: String) (T.unpack . serializeSQL) maybeQuery,
      "elasticsearch" .= maybe (object []) serializeElasticsearch maybeQuery,
      "errors" .= nub errors
    ]

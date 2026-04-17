module Main where

import Data.Aeson (encode)
import qualified Data.ByteString.Lazy.Char8 as BL
import System.Environment (getArgs)
import Themis.Query (parseQueryText, payloadFromQuery)

main :: IO ()
main = do
  args <- getArgs
  let input = unwords args
  case parseQueryText input of
    Left err -> BL.putStrLn (encode (payloadFromQuery input Nothing [err]))
    Right query -> BL.putStrLn (encode (payloadFromQuery input (Just query) []))

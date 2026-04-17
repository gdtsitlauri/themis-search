from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r'"[^"]+"|\(|\)|\^|NEAR/\d+|AND|OR|NOT|[A-Za-z0-9_\-]+(?:\.[0-9]+)?')


@dataclass(slots=True)
class QuerySyntaxError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def tokenize_query(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> dict[str, Any]:
        if not self.tokens:
            return {"type": "RawQuery", "tokens": []}
        expr = self.parse_or()
        if self.pos != len(self.tokens):
            raise QuerySyntaxError(f"Unexpected token: {self.tokens[self.pos]}")
        return optimize_ast(expr)

    def parse_or(self) -> dict[str, Any]:
        node = self.parse_and()
        while self._peek_upper() == "OR":
            self._advance()
            node = {"type": "Or", "left": node, "right": self.parse_and()}
        return node

    def parse_and(self) -> dict[str, Any]:
        node = self.parse_not()
        while True:
            peek = self._peek_upper()
            if peek == "AND":
                self._advance()
            elif peek in {"", "OR", ")"}:
                break
            node = {"type": "And", "left": node, "right": self.parse_not()}
        return node

    def parse_not(self) -> dict[str, Any]:
        if self._peek_upper() == "NOT":
            self._advance()
            return {"type": "Not", "query": self.parse_factor()}
        return self.parse_factor()

    def parse_factor(self) -> dict[str, Any]:
        token = self._peek()
        if token is None:
            raise QuerySyntaxError("Unexpected end of query")
        if token == "(":
            self._advance()
            node = self.parse_or()
            if self._peek() != ")":
                raise QuerySyntaxError("Missing closing parenthesis")
            self._advance()
            return self._maybe_boost(node)
        if token.startswith('"') and token.endswith('"'):
            self._advance()
            node = {"type": "Phrase", "terms": token.strip('"').lower().split()}
            if self._peek_upper().startswith("NEAR/"):
                near_token = self._advance()
                right = self._parse_term_like()
                node = {
                    "type": "Near",
                    "left": " ".join(node["terms"]),
                    "right": right,
                    "distance": int(near_token.split("/", 1)[1]),
                }
            return self._maybe_boost(node)
        node = self._parse_term_like_node()
        if self._peek_upper().startswith("NEAR/"):
            near_token = self._advance()
            right = self._parse_term_like()
            node = {
                "type": "Near",
                "left": node["value"],
                "right": right,
                "distance": int(near_token.split("/", 1)[1]),
            }
        return self._maybe_boost(node)

    def _maybe_boost(self, node: dict[str, Any]) -> dict[str, Any]:
        if self._peek() == "^":
            self._advance()
            token = self._advance()
            if token is None:
                raise QuerySyntaxError("Expected boost weight after ^")
            try:
                weight = float(token)
            except ValueError as exc:
                raise QuerySyntaxError(f"Invalid boost weight: {token}") from exc
            return {"type": "Boost", "query": node, "weight": weight}
        return node

    def _parse_term_like_node(self) -> dict[str, Any]:
        token = self._advance()
        if token is None:
            raise QuerySyntaxError("Expected term")
        upper = token.upper()
        if upper in {"AND", "OR", "NOT"}:
            raise QuerySyntaxError(f"Unexpected operator: {token}")
        return {"type": "Term", "value": token.lower()}

    def _parse_term_like(self) -> str:
        token = self._advance()
        if token is None:
            raise QuerySyntaxError("Expected term after NEAR")
        if token.startswith('"') and token.endswith('"'):
            return token.strip('"').lower()
        return token.lower()

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _peek_upper(self) -> str:
        token = self._peek()
        return token.upper() if token is not None else ""

    def _advance(self) -> str | None:
        token = self._peek()
        if token is not None:
            self.pos += 1
        return token


def parse_query_text(text: str) -> dict[str, Any]:
    parser = _Parser(tokenize_query(text))
    return parser.parse()


def optimize_ast(ast: dict[str, Any]) -> dict[str, Any]:
    ast_type = ast.get("type")
    if ast_type == "Not":
        inner = optimize_ast(ast["query"])
        if inner.get("type") == "Not":
            return optimize_ast(inner["query"])
        if inner.get("type") == "And":
            return optimize_ast({"type": "Or", "left": {"type": "Not", "query": inner["left"]}, "right": {"type": "Not", "query": inner["right"]}})
        if inner.get("type") == "Or":
            return optimize_ast({"type": "And", "left": {"type": "Not", "query": inner["left"]}, "right": {"type": "Not", "query": inner["right"]}})
        return {"type": "Not", "query": inner}
    if ast_type in {"And", "Or"}:
        left = optimize_ast(ast["left"])
        right = optimize_ast(ast["right"])
        if left == right:
            return left
        return {"type": ast_type, "left": left, "right": right}
    if ast_type == "Boost":
        inner = optimize_ast(ast["query"])
        if ast.get("weight", 1.0) == 1.0:
            return inner
        return {"type": "Boost", "query": inner, "weight": ast["weight"]}
    return ast


def ast_to_query_text(ast: dict[str, Any]) -> str:
    ast_type = ast.get("type")
    if ast_type == "Term":
        return ast["value"]
    if ast_type == "Phrase":
        return '"' + " ".join(ast["terms"]) + '"'
    if ast_type == "Near":
        return f"{ast['left']} NEAR/{ast['distance']} {ast['right']}"
    if ast_type == "Boost":
        return f"{ast_to_query_text(ast['query'])}^{ast['weight']}"
    if ast_type == "Not":
        return f"NOT {ast_to_query_text(ast['query'])}"
    if ast_type in {"And", "Or"}:
        return f"({ast_to_query_text(ast['left'])} {ast_type.upper()} {ast_to_query_text(ast['right'])})"
    return " ".join(ast.get("tokens", []))


def ast_to_sql(ast: dict[str, Any]) -> str:
    ast_type = ast.get("type")
    if ast_type == "Term":
        return f"term = '{ast['value']}'"
    if ast_type == "Phrase":
        return f"phrase = '{' '.join(ast['terms'])}'"
    if ast_type == "Near":
        return f"near('{ast['left']}', '{ast['right']}', {ast['distance']})"
    if ast_type == "Boost":
        return f"BOOST({ast_to_sql(ast['query'])}, {ast['weight']})"
    if ast_type == "Not":
        return f"NOT ({ast_to_sql(ast['query'])})"
    if ast_type in {"And", "Or"}:
        return f"({ast_to_sql(ast['left'])} {ast_type.upper()} {ast_to_sql(ast['right'])})"
    return "TRUE"


def ast_to_elasticsearch(ast: dict[str, Any]) -> dict[str, Any]:
    ast_type = ast.get("type")
    if ast_type == "Term":
        return {"term": {"text": ast["value"]}}
    if ast_type == "Phrase":
        return {"match_phrase": {"text": " ".join(ast["terms"])}}
    if ast_type == "Near":
        return {
            "span_near": {
                "clauses": [{"span_term": {"text": ast["left"]}}, {"span_term": {"text": ast["right"]}}],
                "slop": ast["distance"],
            }
        }
    if ast_type == "Boost":
        return {"function_score": {"query": ast_to_elasticsearch(ast["query"]), "boost": ast["weight"]}}
    if ast_type == "Not":
        return {"bool": {"must_not": [ast_to_elasticsearch(ast["query"])]}}
    if ast_type == "And":
        return {"bool": {"must": [ast_to_elasticsearch(ast["left"]), ast_to_elasticsearch(ast["right"])]}}
    if ast_type == "Or":
        return {"bool": {"should": [ast_to_elasticsearch(ast["left"]), ast_to_elasticsearch(ast["right"])]}}
    return {}


def estimate_selectivity(ast: dict[str, Any]) -> float:
    ast_type = ast.get("type")
    if ast_type == "Term":
        return 0.2
    if ast_type == "Phrase":
        return max(0.02, 0.08 / max(len(ast.get("terms", [])), 1))
    if ast_type == "Near":
        return max(0.02, 0.18 / max(int(ast.get("distance", 1)), 1))
    if ast_type == "Boost":
        return estimate_selectivity(ast["query"])
    if ast_type == "Not":
        return max(0.0, min(1.0, 1.0 - estimate_selectivity(ast["query"])))
    if ast_type == "And":
        left = estimate_selectivity(ast["left"])
        right = estimate_selectivity(ast["right"])
        return max(0.0, min(1.0, left * right))
    if ast_type == "Or":
        left = estimate_selectivity(ast["left"])
        right = estimate_selectivity(ast["right"])
        return max(0.0, min(1.0, left + right - left * right))
    return 1.0


def payload_from_ast(raw: str, ast: dict[str, Any], errors: list[str] | None = None) -> dict[str, Any]:
    normalized = ast_to_query_text(ast)
    return {
        "raw": raw,
        "normalized": normalized,
        "ast": ast,
        "estimated_selectivity": round(estimate_selectivity(ast), 4),
        "sql": ast_to_sql(ast),
        "elasticsearch": ast_to_elasticsearch(ast),
        "errors": list(errors or []),
    }


def normalize_external_ast(ast: Any) -> dict[str, Any]:
    if not isinstance(ast, dict):
        return {"type": "RawQuery", "tokens": []}
    if "type" in ast:
        return ast
    tag = ast.get("tag")
    contents = ast.get("contents")
    if tag == "Term":
        return {"type": "Term", "value": str(contents).lower()}
    if tag == "Phrase":
        return {"type": "Phrase", "terms": [str(item).lower() for item in contents or []]}
    if tag == "Near" and isinstance(contents, list) and len(contents) == 3:
        return {
            "type": "Near",
            "left": str(contents[0]).lower(),
            "right": str(contents[1]).lower(),
            "distance": int(contents[2]),
        }
    if tag == "Boost" and isinstance(contents, list) and len(contents) == 2:
        return {"type": "Boost", "query": normalize_external_ast(contents[0]), "weight": float(contents[1])}
    if tag in {"And", "Or"} and isinstance(contents, list) and len(contents) == 2:
        return {
            "type": tag,
            "left": normalize_external_ast(contents[0]),
            "right": normalize_external_ast(contents[1]),
        }
    if tag == "Not":
        return {"type": "Not", "query": normalize_external_ast(contents)}
    return {"type": "RawQuery", "tokens": ast.get("tokens", [])}

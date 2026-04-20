"""Computational complexity demonstrations for the THEMIS theory track.

Covers P vs NP, NP-complete problems (SAT, Subset Sum, Graph Coloring),
and tractable baselines.
"""
from __future__ import annotations

import itertools
import time
from typing import Any


# ---------------------------------------------------------------------------
# P-class: polynomial-time algorithms
# ---------------------------------------------------------------------------

def solve_2sat(clauses: list[tuple[int, int]], n_vars: int) -> dict[str, Any]:
    """Solve 2-SAT in O(n+m) via Kosaraju's SCC. Variables 1..n_vars."""
    # Build implication graph: -x means NOT x (offset by n_vars)
    def var(x: int) -> int:
        return x - 1 if x > 0 else n_vars + (-x) - 1

    graph: list[list[int]] = [[] for _ in range(2 * n_vars)]
    rgraph: list[list[int]] = [[] for _ in range(2 * n_vars)]

    for a, b in clauses:
        # (a OR b) == (NOT a -> b) AND (NOT b -> a)
        na, nb = -a, -b
        graph[var(na)].append(var(b))
        graph[var(nb)].append(var(a))
        rgraph[var(b)].append(var(na))
        rgraph[var(a)].append(var(nb))

    visited = [False] * 2 * n_vars
    order: list[int] = []

    def dfs1(u: int) -> None:
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    def dfs2(u: int, comp: int) -> None:
        comp_id[u] = comp
        for v in rgraph[u]:
            if comp_id[v] == -1:
                dfs2(v, comp)

    for i in range(2 * n_vars):
        if not visited[i]:
            dfs1(i)

    comp_id = [-1] * 2 * n_vars
    comp = 0
    for u in reversed(order):
        if comp_id[u] == -1:
            dfs2(u, comp)
            comp += 1

    satisfiable = all(comp_id[var(i)] != comp_id[var(-i)] for i in range(1, n_vars + 1))
    assignment = {}
    if satisfiable:
        for i in range(1, n_vars + 1):
            assignment[f"x{i}"] = comp_id[var(i)] > comp_id[var(-i)]
    return {
        "problem": "2-SAT",
        "class": "P",
        "n_vars": n_vars,
        "n_clauses": len(clauses),
        "satisfiable": satisfiable,
        "assignment": assignment,
    }


def solve_bipartite_matching(graph: dict[str, list[str]]) -> dict[str, Any]:
    """Maximum bipartite matching via augmenting paths (Hopcroft-Karp sketch)."""
    left = list(graph.keys())
    match_l: dict[str, str | None] = {u: None for u in left}
    match_r: dict[str, str | None] = {}

    def dfs(u: str, visited: set[str]) -> bool:
        for v in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                if match_r.get(v) is None or dfs(match_r[v], visited):  # type: ignore[arg-type]
                    match_l[u] = v
                    match_r[v] = u
                    return True
        return False

    matching = 0
    for u in left:
        if dfs(u, set()):
            matching += 1

    return {
        "problem": "Bipartite Matching",
        "class": "P",
        "left_nodes": len(left),
        "right_nodes": len(set(v for vs in graph.values() for v in vs)),
        "max_matching": matching,
    }


# ---------------------------------------------------------------------------
# NP-complete: exhaustive / bounded solvers
# ---------------------------------------------------------------------------

def solve_3sat(clauses: list[tuple[int, int, int]], n_vars: int) -> dict[str, Any]:
    """Solve 3-SAT by exhaustive search (NP-complete; exponential in n_vars)."""
    start = time.perf_counter()

    def check(assignment: dict[int, bool]) -> bool:
        for clause in clauses:
            satisfied = False
            for lit in clause:
                val = assignment[abs(lit)]
                if lit < 0:
                    val = not val
                if val:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

    for bits in itertools.product([False, True], repeat=n_vars):
        assignment = {i + 1: v for i, v in enumerate(bits)}
        if check(assignment):
            elapsed = time.perf_counter() - start
            return {
                "problem": "3-SAT",
                "class": "NP-complete",
                "n_vars": n_vars,
                "n_clauses": len(clauses),
                "satisfiable": True,
                "assignment": {f"x{k}": v for k, v in assignment.items()},
                "assignments_checked": 2 ** n_vars,
                "elapsed_seconds": elapsed,
            }

    return {
        "problem": "3-SAT",
        "class": "NP-complete",
        "n_vars": n_vars,
        "n_clauses": len(clauses),
        "satisfiable": False,
        "assignment": {},
        "assignments_checked": 2 ** n_vars,
        "elapsed_seconds": time.perf_counter() - start,
    }


def solve_subset_sum(numbers: list[int], target: int) -> dict[str, Any]:
    """Solve Subset Sum by exhaustive search (NP-complete in general).
    Uses DP when n <= 30 for efficiency."""
    start = time.perf_counter()
    n = len(numbers)

    if n <= 30:
        # DP approach: O(n * target)
        dp = [False] * (target + 1)
        dp[0] = True
        parent: list[list[int | None]] = [[None] * (target + 1) for _ in range(n + 1)]
        for i, num in enumerate(numbers):
            for t in range(target, num - 1, -1):
                if dp[t - num] and not dp[t]:
                    dp[t] = True
                    parent[i + 1][t] = t - num
            for t in range(target + 1):
                if parent[i + 1][t] is None and dp[t]:
                    parent[i + 1][t] = -1

        found = dp[target]
        subset: list[int] = []
        if found:
            t = target
            for i in range(n, 0, -1):
                p = parent[i][t]
                if p is not None and p != -1 and p != t:
                    subset.append(numbers[i - 1])
                    t = p
    else:
        found = False
        subset = []

    return {
        "problem": "Subset Sum",
        "class": "NP-complete",
        "n": n,
        "target": target,
        "solvable": found,
        "subset": subset,
        "elapsed_seconds": time.perf_counter() - start,
    }


def solve_graph_coloring(
    nodes: list[str],
    edges: list[tuple[str, str]],
    k: int,
) -> dict[str, Any]:
    """k-graph-coloring by backtracking (NP-complete for k >= 3)."""
    start = time.perf_counter()
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    coloring: dict[str, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            return True
        node = nodes[idx]
        used = {coloring[nb] for nb in adj[node] if nb in coloring}
        for color in range(k):
            if color not in used:
                coloring[node] = color
                if backtrack(idx + 1):
                    return True
                del coloring[node]
        return False

    solvable = backtrack(0)
    return {
        "problem": f"{k}-Graph Coloring",
        "class": "NP-complete" if k >= 3 else "P",
        "nodes": len(nodes),
        "edges": len(edges),
        "k": k,
        "colorable": solvable,
        "coloring": {n: coloring.get(n) for n in nodes} if solvable else {},
        "elapsed_seconds": time.perf_counter() - start,
    }


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_complexity_suite() -> dict[str, Any]:
    """Run all complexity demos and return structured results."""
    results: list[dict[str, Any]] = []

    # P: 2-SAT
    # (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
    r = solve_2sat([(1, 2), (-1, 3), (-2, -3)], n_vars=3)
    results.append(r)

    # P: Bipartite Matching
    r = solve_bipartite_matching({"a": ["1", "2"], "b": ["2", "3"], "c": ["3"]})
    results.append(r)

    # NP-complete: 3-SAT (satisfiable, 4 vars)
    r = solve_3sat([(1, 2, -3), (-1, 3, 4), (2, -3, -4), (-2, -1, 3)], n_vars=4)
    results.append(r)

    # NP-complete: 3-SAT (unsatisfiable: x AND NOT x AND ...)
    r = solve_3sat([(1, 1, 1), (-1, -1, -1)], n_vars=1)
    results.append(r)

    # NP-complete: Subset Sum
    r = solve_subset_sum([3, 1, 4, 1, 5, 9, 2, 6], target=11)
    results.append(r)

    r = solve_subset_sum([7, 13, 11, 17], target=30)
    results.append(r)

    # NP-complete: Graph Coloring
    # Triangle graph: needs 3 colors
    r = solve_graph_coloring(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C"), ("A", "C")],
        k=2,
    )
    results.append(r)

    r = solve_graph_coloring(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C"), ("A", "C")],
        k=3,
    )
    results.append(r)

    # Petersen graph (3-chromatic)
    petersen_nodes = [str(i) for i in range(10)]
    petersen_edges = [
        ("0","1"),("1","2"),("2","3"),("3","4"),("4","0"),
        ("5","7"),("7","9"),("9","6"),("6","8"),("8","5"),
        ("0","5"),("1","6"),("2","7"),("3","8"),("4","9"),
    ]
    r = solve_graph_coloring(petersen_nodes, petersen_edges, k=3)
    results.append(r)

    return {
        "suite": "complexity",
        "results": results,
        "summary": [
            {
                "problem": r["problem"],
                "class": r["class"],
                "solved": r.get("satisfiable", r.get("solvable", r.get("colorable", r.get("max_matching")))),
            }
            for r in results
        ],
    }

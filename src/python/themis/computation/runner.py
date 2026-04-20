"""Computation theory suite runner — generates results tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from themis.computation.automata import DFA, NFA
from themis.computation.turing import TuringMachine
from themis.computation.complexity import run_complexity_suite


def run_automata_suite() -> dict[str, Any]:
    dfa_cases = [
        ("even_zeros", DFA.even_zeros(), ["0", "1", "00", "01", "10", "11", "000", "001", "100", "110"]),
        ("ends_with_ab", DFA.ends_with_ab(), ["ab", "aab", "b", "aba", "bab", "abab", ""]),
        ("divisible_by_3", DFA.divisible_by_3(), ["0","1","10","11","100","101","110","111","1001"]),
    ]
    dfa_rows = []
    for name, dfa, words in dfa_cases:
        for w in words:
            dfa_rows.append({"automaton": name, "word": w or "(empty)", "accepted": dfa.accepts(w)})

    nfa = NFA.contains_ab()
    nfa_dfa = nfa.to_dfa()
    nfa_words = ["ab", "aab", "ba", "aabb", "b", "a", "abba", "bab"]
    nfa_rows = []
    for w in nfa_words:
        nfa_rows.append({
            "word": w,
            "nfa_accepts": nfa.accepts(w),
            "converted_dfa_accepts": nfa_dfa.accepts(w),
            "consistent": nfa.accepts(w) == nfa_dfa.accepts(w),
        })

    eps_nfa = NFA.epsilon_demo()
    eps_dfa = eps_nfa.to_dfa()
    eps_words = ["", "a", "b", "aa", "ab", "aab", "abb", "b", "bb"]
    eps_rows = []
    for w in eps_words:
        eps_rows.append({
            "word": w or "(empty)",
            "nfa_accepts": eps_nfa.accepts(w),
            "dfa_accepts": eps_dfa.accepts(w),
            "consistent": eps_nfa.accepts(w) == eps_dfa.accepts(w),
        })

    return {
        "dfa_results": dfa_rows,
        "nfa_subset_construction": nfa_rows,
        "epsilon_closure": eps_rows,
    }


def run_turing_suite() -> list[dict[str, Any]]:
    cases = [
        ("palindrome", TuringMachine.palindrome_checker(), ["", "0", "1", "00", "11", "01", "010", "101", "001", "0110"]),
        ("increment", TuringMachine.increment_binary(), ["0", "1", "10", "11", "101", "111", "1001"]),
        ("copy", TuringMachine.copy_string(), ["1", "11", "111", "1111"]),
    ]
    rows = []
    for name, tm, inputs in cases:
        for w in inputs:
            result = tm.run(w)
            rows.append({
                "machine": name,
                "input": w or "(empty)",
                "accepted": result["accepted"],
                "steps": result["steps"],
                "final_tape": result["tape"],
            })
    return rows


def save_results(output_dir: str = "results/computation") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    automata = run_automata_suite()
    turing = run_turing_suite()
    complexity = run_complexity_suite()

    for name, rows in [
        ("dfa_results", automata["dfa_results"]),
        ("nfa_subset_construction", automata["nfa_subset_construction"]),
        ("epsilon_closure", automata["epsilon_closure"]),
        ("turing_results", turing),
    ]:
        if rows:
            keys = list(rows[0].keys())
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    with open(out / "complexity_results.json", "w") as f:
        json.dump(complexity, f, indent=2, default=str)

    summary = {
        "dfa_tests": len(automata["dfa_results"]),
        "nfa_consistency_checks": all(r["consistent"] for r in automata["nfa_subset_construction"]),
        "epsilon_consistency_checks": all(r["consistent"] for r in automata["epsilon_closure"]),
        "turing_runs": len(turing),
        "complexity_demos": len(complexity["results"]),
        "p_vs_np_examples": complexity["summary"],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Computation theory results saved to {out}/")
    print(f"  DFA tests: {summary['dfa_tests']}")
    print(f"  NFA->DFA consistent: {summary['nfa_consistency_checks']}")
    print(f"  eps-closure consistent: {summary['epsilon_consistency_checks']}")
    print(f"  Turing machine runs: {summary['turing_runs']}")
    print(f"  Complexity demos: {summary['complexity_demos']}")


if __name__ == "__main__":
    save_results()

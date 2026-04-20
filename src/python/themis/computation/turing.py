"""Turing Machine implementation for the THEMIS computation theory track."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BLANK = "_"


@dataclass
class TuringMachine:
    """Single-tape deterministic Turing Machine.

    Parameters
    ----------
    states: set of state labels
    alphabet: tape alphabet (includes BLANK '_')
    transitions: dict (state, symbol) -> (new_state, write_symbol, direction)
                 direction is 'L' (left) or 'R' (right)
    start: initial state
    accept: accepting (halting) state
    reject: rejecting (halting) state
    max_steps: safety limit to detect infinite loops
    """

    states: set[str]
    alphabet: set[str]
    transitions: dict[tuple[str, str], tuple[str, str, str]]
    start: str
    accept: str
    reject: str
    max_steps: int = 10_000

    def run(self, input_word: str) -> dict[str, Any]:
        """Run the TM on *input_word* and return result dict."""
        tape: list[str] = list(input_word) if input_word else [BLANK]
        head = 0
        state = self.start
        steps = 0
        trace: list[dict[str, Any]] = []

        while state not in {self.accept, self.reject} and steps < self.max_steps:
            if head < 0:
                tape.insert(0, BLANK)
                head = 0
            if head >= len(tape):
                tape.append(BLANK)

            symbol = tape[head]
            trace.append({
                "step": steps,
                "state": state,
                "head": head,
                "tape": "".join(tape),
                "symbol": symbol,
            })

            key = (state, symbol)
            if key not in self.transitions:
                state = self.reject
                break

            new_state, write, direction = self.transitions[key]
            tape[head] = write
            state = new_state
            head += 1 if direction == "R" else -1
            steps += 1

        tape_str = "".join(tape).strip(BLANK) or BLANK
        return {
            "accepted": state == self.accept,
            "halted": state in {self.accept, self.reject},
            "steps": steps,
            "final_state": state,
            "tape": tape_str,
            "trace_length": len(trace),
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def palindrome_checker(cls) -> "TuringMachine":
        """TM that accepts binary palindromes over {0,1}."""
        return cls(
            states={"q0","q1","q2","q3","q4","q5","q6","qA","qR"},
            alphabet={"0","1","X","_"},
            transitions={
                ("q0","_"): ("qA","_","R"),
                ("q0","X"): ("q0","X","R"),
                ("q0","0"): ("q1","X","R"),
                ("q0","1"): ("q2","X","R"),
                ("q1","0"): ("q1","0","R"),
                ("q1","1"): ("q1","1","R"),
                ("q1","X"): ("q1","X","R"),
                ("q1","_"): ("q3","_","L"),
                ("q2","0"): ("q2","0","R"),
                ("q2","1"): ("q2","1","R"),
                ("q2","X"): ("q2","X","R"),
                ("q2","_"): ("q4","_","L"),
                ("q3","X"): ("q5","X","L"),
                ("q3","0"): ("q5","X","L"),
                ("q3","1"): ("qR","1","L"),
                ("q4","X"): ("q5","X","L"),
                ("q4","1"): ("q5","X","L"),
                ("q4","0"): ("qR","0","L"),
                ("q5","0"): ("q5","0","L"),
                ("q5","1"): ("q5","1","L"),
                ("q5","X"): ("q0","X","R"),
                ("q5","_"): ("q0","_","R"),
            },
            start="q0",
            accept="qA",
            reject="qR",
        )

    @classmethod
    def increment_binary(cls) -> "TuringMachine":
        """TM that increments a binary number by 1 (rightmost bit first)."""
        return cls(
            states={"q0","q1","qA","qR"},
            alphabet={"0","1","_"},
            transitions={
                ("q0","0"): ("q0","0","R"),
                ("q0","1"): ("q0","1","R"),
                ("q0","_"): ("q1","_","L"),
                ("q1","1"): ("q1","0","L"),
                ("q1","0"): ("qA","1","R"),
                ("q1","_"): ("qA","1","R"),
            },
            start="q0",
            accept="qA",
            reject="qR",
        )

    @classmethod
    def copy_string(cls) -> "TuringMachine":
        """TM that copies a unary string: 111 -> 111_111."""
        return cls(
            states={"q0","q1","q2","q3","q4","qA","qR"},
            alphabet={"1","X","_"},
            transitions={
                ("q0","1"): ("q1","X","R"),
                ("q0","_"): ("qA","_","R"),
                ("q0","X"): ("q4","X","R"),
                ("q1","1"): ("q1","1","R"),
                ("q1","_"): ("q2","_","R"),
                ("q1","X"): ("q1","X","R"),
                ("q2","1"): ("q2","1","R"),
                ("q2","_"): ("q3","1","L"),
                ("q3","1"): ("q3","1","L"),
                ("q3","_"): ("q3","_","L"),
                ("q3","X"): ("q0","X","R"),
                ("q4","X"): ("q4","X","R"),
                ("q4","_"): ("qA","_","R"),
            },
            start="q0",
            accept="qA",
            reject="qR",
        )

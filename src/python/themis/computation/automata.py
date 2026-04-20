"""DFA and NFA implementations for the THEMIS computation theory track."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DFA:
    """Deterministic Finite Automaton.

    Parameters
    ----------
    states: set of state labels
    alphabet: set of input symbols
    transitions: dict mapping (state, symbol) -> state
    start: initial state
    accept: set of accepting states
    """

    states: set[str]
    alphabet: set[str]
    transitions: dict[tuple[str, str], str]
    start: str
    accept: set[str]

    def accepts(self, word: str) -> bool:
        """Return True if *word* is accepted by this DFA."""
        current = self.start
        for symbol in word:
            if symbol not in self.alphabet:
                return False
            key = (current, symbol)
            if key not in self.transitions:
                return False
            current = self.transitions[key]
        return current in self.accept

    def trace(self, word: str) -> list[dict[str, Any]]:
        """Return step-by-step execution trace."""
        steps: list[dict[str, Any]] = []
        current = self.start
        steps.append({"step": 0, "state": current, "remaining": word, "symbol": None})
        for i, symbol in enumerate(word):
            key = (current, symbol)
            next_state = self.transitions.get(key)
            current = next_state if next_state is not None else "__dead__"
            steps.append({
                "step": i + 1,
                "state": current,
                "symbol": symbol,
                "remaining": word[i + 1:],
            })
        steps[-1]["accepted"] = current in self.accept
        return steps

    @classmethod
    def even_zeros(cls) -> "DFA":
        """DFA that accepts binary strings with an even number of 0s."""
        return cls(
            states={"even", "odd"},
            alphabet={"0", "1"},
            transitions={
                ("even", "0"): "odd",
                ("even", "1"): "even",
                ("odd", "0"): "even",
                ("odd", "1"): "odd",
            },
            start="even",
            accept={"even"},
        )

    @classmethod
    def ends_with_ab(cls) -> "DFA":
        """DFA over {a,b} accepting strings that end in 'ab'."""
        return cls(
            states={"q0", "q1", "q2"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): "q1",
                ("q0", "b"): "q0",
                ("q1", "a"): "q1",
                ("q1", "b"): "q2",
                ("q2", "a"): "q1",
                ("q2", "b"): "q0",
            },
            start="q0",
            accept={"q2"},
        )

    @classmethod
    def divisible_by_3(cls) -> "DFA":
        """DFA over {0,1} accepting binary representations divisible by 3."""
        return cls(
            states={"r0", "r1", "r2"},
            alphabet={"0", "1"},
            transitions={
                ("r0", "0"): "r0",
                ("r0", "1"): "r1",
                ("r1", "0"): "r2",
                ("r1", "1"): "r0",
                ("r2", "0"): "r1",
                ("r2", "1"): "r2",
            },
            start="r0",
            accept={"r0"},
        )


@dataclass
class NFA:
    """Nondeterministic Finite Automaton with epsilon (ε) transitions.

    Epsilon transitions are represented by the symbol '' (empty string).
    """

    states: set[str]
    alphabet: set[str]
    transitions: dict[tuple[str, str], set[str]]
    start: str
    accept: set[str]

    def _epsilon_closure(self, states: set[str]) -> set[str]:
        """Compute ε-closure of a set of states."""
        closure = set(states)
        queue = deque(states)
        while queue:
            s = queue.popleft()
            for t in self.transitions.get((s, ""), set()):
                if t not in closure:
                    closure.add(t)
                    queue.append(t)
        return closure

    def accepts(self, word: str) -> bool:
        """Return True if *word* is accepted."""
        current = self._epsilon_closure({self.start})
        for symbol in word:
            if symbol not in self.alphabet:
                return False
            next_states: set[str] = set()
            for state in current:
                next_states |= self.transitions.get((state, symbol), set())
            current = self._epsilon_closure(next_states)
        return bool(current & self.accept)

    def to_dfa(self) -> DFA:
        """Subset construction: convert NFA to equivalent DFA."""
        start_closure = frozenset(self._epsilon_closure({self.start}))
        dfa_states: dict[frozenset[str], str] = {}
        counter = 0

        def label(fs: frozenset[str]) -> str:
            nonlocal counter
            if fs not in dfa_states:
                dfa_states[fs] = f"D{counter}"
                counter += 1
            return dfa_states[fs]

        label(start_closure)
        queue: deque[frozenset[str]] = deque([start_closure])
        dfa_transitions: dict[tuple[str, str], str] = {}
        visited: set[frozenset[str]] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for symbol in self.alphabet:
                next_states: set[str] = set()
                for s in current:
                    next_states |= self.transitions.get((s, symbol), set())
                closure = frozenset(self._epsilon_closure(next_states))
                label(closure)
                dfa_transitions[(label(current), symbol)] = label(closure)
                if closure not in visited:
                    queue.append(closure)

        dfa_accept = {label(fs) for fs in dfa_states if fs & self.accept}
        return DFA(
            states=set(dfa_states.values()),
            alphabet=self.alphabet,
            transitions=dfa_transitions,
            start=label(start_closure),
            accept=dfa_accept,
        )

    @classmethod
    def contains_ab(cls) -> "NFA":
        """NFA over {a,b} accepting strings containing 'ab' as substring."""
        return cls(
            states={"q0", "q1", "q2"},
            alphabet={"a", "b"},
            transitions={
                ("q0", "a"): {"q0", "q1"},
                ("q0", "b"): {"q0"},
                ("q1", "b"): {"q2"},
                ("q2", "a"): {"q2"},
                ("q2", "b"): {"q2"},
            },
            start="q0",
            accept={"q2"},
        )

    @classmethod
    def epsilon_demo(cls) -> "NFA":
        """NFA with ε-transitions accepting {a^n b^m : n,m >= 0}."""
        return cls(
            states={"s0", "s1", "s2"},
            alphabet={"a", "b"},
            transitions={
                ("s0", "a"): {"s0"},
                ("s0", ""): {"s1"},
                ("s1", "b"): {"s2"},
                ("s2", "b"): {"s2"},
            },
            start="s0",
            accept={"s1", "s2"},
        )

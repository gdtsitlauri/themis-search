# THEMIS


Trustworthy Hybrid Engine for Multimodal Intelligent Search

THEMIS is a research-oriented multimodal search and media-analysis framework that combines:
- classical information retrieval
- neural and hybrid retrieval
- a typed Haskell query language
- multimedia forensics
- causal reranking through THEMIS-ORACLE

The project is designed as a reproducible research prototype rather than a thin demo. It includes runnable code, passing tests, committed benchmark artifacts, and a paper draft with real numbers derived from the repository outputs.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Primary Research Thesis

The most important and most defensible reading of THEMIS is:

- a **hybrid retrieval and reranking research platform**,
- centered on **THEMIS-ORACLE** as the repository's main novel algorithmic
  identity,
- with multimedia forensics and computation-theory modules included as
  technically real supporting research tracks.

This framing is stronger than presenting THEMIS as five equal sub-projects at
once. The repository's clearest empirical center of gravity is the retrieval +
ORACLE stack.

## Core idea

Most search systems estimate relevance through lexical overlap or embedding similarity.

THEMIS extends that idea with **causal relevance**:
- not only “this document looks similar to the query”
- but also “this document appears upstream of the answer and helps explain why it matters”

The same philosophy is applied to media forensics:
- not only “this image looks manipulated”
- but also “this image was likely manipulated because these artifacts co-occur and form a causal chain”

## Repository layout

- `src/python/themis/`: orchestration, IR, neural retrieval, hybrid search, API, forensics, ORACLE, benchmarks
- `src/haskell/`: query DSL, parser, optimizer, serializers, semantics-oriented tests
- `src/scala/`: inverted index, Boolean and positional retrieval, Spark-oriented indexing hooks
- `data/fixtures/`: tiny committed corpora, local manifests, and forensic fixtures
- `results/`: committed benchmark outputs and example explanation artifacts
- `paper/`: research paper draft
- `tests/`: Python integration tests and root Haskell spec mirror

## Implemented research modules

- Classical IR: inverted index, positional index, TF-IDF, BM25, Dirichlet LM, PageRank
- Query execution: Boolean retrieval, phrase queries, NEAR/proximity queries, boosted queries
- Neural search: dense retrieval, FAISS exact/IVF paths, cross-encoder reranking, ColBERT-style late interaction, hybrid fusion
- Query language: typed AST, parser combinators, optimizer, selectivity estimation, JSON/SQL/Elasticsearch serialization
- Forensics: image copy-move and frequency heuristics, video frame-duplication checks, audio splice/background inconsistency checks
- Causal reasoning: PC-style causal skeleton discovery, ORACLE reranking, counterfactual result explanations, forensic causal chains
- API/CLI: FastAPI search surface plus CLI commands for search, indexing, forensics, benchmarking, dataset status, and external evaluation hooks

## Verification status

- Python integration suite: `11/11` passing
- Haskell package tests: `cabal test` passing
- Scala tests: `sbt test` passing
- Search API latency on fixture corpus: under `100 ms`
- Dense latency at 1000 docs: `6.6027 ms`
- Forensics batch throughput: `81.894 img/s`
- ORACLE mean top score: `1.3324`
- ORACLE changes the dense top result on `94.44%` of synthetic benchmark queries

## Retrieval and ORACLE Snapshot

Key committed retrieval-side artifacts:

- `results/ir/ranking_comparison.csv`
- `results/neural_search/latency_benchmark.csv`
- `results/themis_oracle/causal_vs_standard.csv`

Representative committed numbers:

- BM25 mean MAP across 3 seeds: about `0.483`
- Dense latency at 1000 docs: about `6.60 ms`
- ORACLE mean top score: `1.3324`
- ORACLE changes the dense top result on `94.44%` of synthetic benchmark queries

Interpretation:

- The main research value is not raw dense retrieval alone.
- It is the combination of sparse + dense retrieval with a causal reranking
  layer that materially changes the final top result distribution.

## Why THEMIS Can Now Be Read More Strongly

THEMIS is closer to the stronger bucket when it is foregrounded as a retrieval
and reranking system with real supporting depth:

- the core retrieval story already has committed ranking and latency artifacts,
- `THEMIS-ORACLE` gives the repository a clearer algorithmic identity than a
  generic multimodal-search wrapper,
- the forensics and computation modules now reinforce the repository's research
  breadth instead of diluting its main thesis.

## Current artifacts

Main committed outputs:
- `results/ir/retrieval_benchmarks.csv`
- `results/ir/ranking_comparison.csv`
- `results/neural_search/dense_vs_sparse.csv`
- `results/neural_search/latency_benchmark.csv`
- `results/forensics/image_detection_results.csv`
- `results/forensics/causal_forensics_chain.json`
- `results/themis_oracle/causal_vs_standard.csv`
- `results/themis_oracle/explanations_sample.json`

## Evidence Hierarchy

THEMIS is broad by design, but the repository should be interpreted as a
**research platform with multiple validated modules**, not as a single narrow
benchmark paper.

- Primary evidence:
  - retrieval benchmarks
  - neural-search latency artifacts
  - THEMIS-ORACLE reranking outputs
- Secondary evidence:
  - multimedia forensics detection and causal-chain artifacts
- Supporting research/education evidence:
  - computation-theory module and formal language machinery

This makes THEMIS valuable and technically distinctive, but somewhat more
diffuse than the most tightly focused flagship repos.

## Why THEMIS is still close to the upper tier

- The search, neural-retrieval, and forensics layers all have committed output
  artifacts rather than only architectural claims.
- The multi-language implementation is real and test-backed across Python,
  Haskell, and Scala.
- The repository is broad, but it is not decorative breadth; multiple modules
  are implemented and exercised.
- What keeps THEMIS slightly below the most confident flagship bucket is not
  lack of value, but the fact that its methodological center is spread across
  several sub-research tracks instead of one sharply bounded empirical thesis.

## Strongest empirical sub-story

If THEMIS is foregrounded, the clearest and strongest story is:

- hybrid and neural retrieval with committed latency/ranking artifacts,
- plus THEMIS-ORACLE causal reranking backed by
  `results/themis_oracle/causal_vs_standard.csv`,
- with forensics and computation-theory modules presented as valuable
  supporting research tracks rather than as the single main claim.

## What Still Keeps THEMIS Slightly Below The Very Top Tier

- the repository still spans multiple research directions at once,
- the retrieval + ORACLE story is stronger than the forensics side
  empirically,
- the next real jump would come from larger external collections and broader
  committed benchmark diversity.

Optional local-manifest summaries are also supported and already demonstrated through fixture-backed examples:
- `results/ir/external_collection_summary.json`
- `results/neural_search/external_collection_summary.json`
- `results/forensics/external_dataset_summary.json`
- `results/themis_oracle/external_collection_summary.json`

## Quick start

Run Python tests:

```bash
PYTHONPATH=src/python python3 -m unittest discover -s tests -v
```

Run Haskell tests:

```bash
cd src/haskell && cabal test
```

Run Scala tests:

```bash
cd src/scala && sbt test
```

Run the benchmark pipeline:

```bash
PYTHONPATH=src/python python3 -m themis.cli benchmark
```

Inspect dataset availability configured in `configs/app.json`:

```bash
PYTHONPATH=src/python python3 -m themis.cli dataset-status
```

Run local external-manifest benchmarks:

```bash
PYTHONPATH=src/python python3 -m themis.cli benchmark-external --ir-manifest data/fixtures/external_ir_manifest.json --forensics-manifest data/fixtures/external_forensics_manifest.json
```

Run a search query:

```bash
PYTHONPATH=src/python python3 -m themis.cli search --query 'causal relevance AND "counterfactual ranking"'
```

Inspect index stats:

```bash
PYTHONPATH=src/python python3 -m themis.cli index
```

Warm local neural models:

```bash
PYTHONPATH=src/python python3 -m themis.cli warm-models
```

Run a forensics sample:

```bash
PYTHONPATH=src/python python3 -m themis.cli forensics --image data/fixtures/forensics_image_fake.json
```

## Supporting Computation Theory Module

`src/python/themis/computation/` adds a formal computation theory track covering three pillars of theoretical computer science:

### Finite Automata (`automata.py`)

| Class | Highlights |
|---|---|
| `DFA` | `accepts()`, `trace()`, factory methods: `even_zeros`, `ends_with_ab`, `divisible_by_3` |
| `NFA` | ε-closure, `accepts()` via powerset simulation, `to_dfa()` subset construction |

### Turing Machines (`turing.py`)

| Machine | Description |
|---|---|
| `palindrome_checker` | Two-pointer crossing scan |
| `increment_binary` | Ripple carry from rightmost bit |
| `copy_string` | Mark-copy-return loop |

### Complexity Classes (`complexity.py`)

| Problem | Algorithm | Class |
|---|---|---|
| 2-SAT | Kosaraju SCC (O(n+m)) | P |
| Bipartite Matching | Augmenting paths | P |
| 3-SAT | Exhaustive 2^n enumeration | NP-complete |
| Subset Sum | DP O(n·target) | NP-complete |
| k-Graph Coloring | Backtracking | NP-complete |

### Results

```
DFA tests:              26
NFA→DFA consistent:     True
ε-closure consistent:   True
Turing machine runs:    21
Complexity demos:       9
```

Results saved to `results/computation/` (dfa_results.csv, turing_results.csv, complexity_results.json).

Run the suite:

```bash
PYTHONPATH=src/python python3 -m themis.computation.runner
```

## Research notes

- the codebase is multi-language and modular
- the novelty claim is explicit and implemented
- the experiments are reproducible inside the repository
- the paper draft is synchronized with committed metrics
- the retrieval + ORACLE axis provides the clearest empirical thesis for the
  broader platform



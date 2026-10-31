# Guide to Our Additions

What we added on top of the starter, where it lives, and how the pieces fit
together. The starter itself is described in [baseline.md](baseline.md).
Design reasons are in [decisions.md](decisions.md), and results are in
[EXPERIMENTS.md](../EXPERIMENTS.md).

## Principles

- **The starter is edited only deliberately** ([D1](decisions.md#d1-extend-the-starter-through-its-plug-in-points-edit-it-only-deliberately)).
  Our code plugs into it through the `"class"` keys in config files. A
  starter edit adds a general hook, gets its own test, commit and decision
  entry, and is listed under "Starter files changed" below.
- **One strategy per small module.** Trying a new retrieval, chunking or
  embedding idea means adding one class and one config file. Other parts
  stay untouched.
- **Follow the starter's style.** Standard library at runtime where
  practical, type hints, NumPy-style docstrings, short functions, and a
  unit test for each behavior.

## How a Question Flows

```text
question ─► RAGService (starter)            checks scope and output rules
              ├─ retriever.retrieve()       ◄── swapped via config "retriever.class"
              └─ generator.generate()       ◄── swapped via config "generator.class"
```

`RAGService` can itself be swapped via `"service.class"` for steps that span
both stages, e.g. retrying retrieval.

## The Composed Retriever

`ComposedRetriever` replaces the starter's `EmbeddingRetriever` and is put
together from parts, each named in the `retriever` config section
([D3](decisions.md#d3-make-retrieval-a-set-of-small-swappable-strategies)):

```text
indexing (once, when the retriever loads)
  paragraph ─► Chunker.split ─► pieces ─► TextView.render ─► Unit
  all units ─► each Scorer.fit           (BM25 counts, embeddings, ...)

per question
  units of question["paper_ids"] ─► each Scorer.score ─► Fusion.fuse
    ─► best unit per paragraph ─► top candidate_k paragraphs
    ─► Reranker.rerank (optional) ─► top_k original paragraphs
```

| Part | Interface (`retrieval/base.py`) | Implementations | Config key |
| --- | --- | --- | --- |
| Chunker | `split(paragraph) -> list[str]` | `chunkers:WholeParagraph` | `chunker` (default `WholeParagraph`) |
| Text view | `render(paragraph, piece) -> str` | `views:RawText` | `view` (default `RawText`) |
| Scorer | `fit(units)`, `score(query, candidates) -> list[float]` | `dense:DenseScorer` | `scorers` (list, required) |
| Fusion | `fuse(score_lists) -> list[float]` | `fusion:ReciprocalRankFusion` | `fusion` (default RRF, `k` = 60) |
| Reranker | `rerank(question, passages) -> list[passage]` | none yet | `reranker` (optional), with `candidate_k` |

Units always keep their `paragraph_id`, and the retriever returns the
original paragraph dicts. That is why chunking and text views cannot break
the output contract.

`config/composed-dense.json` sets up the baseline's behavior from these
parts (raw text, whole paragraphs, one dense scorer). It is the reference
point for every retrieval experiment.

### Adding a New Strategy

1. Write a class with the interface from the table in the matching module,
   e.g. a `SectionPrefixed` view in `retrieval/views.py`. Read options from
   `settings`.
2. Copy `config/composed-dense.json` to a new config and change only that
   part's `class` (one change per config makes comparisons clean).
3. Add a unit test in `tests/test_composed_retriever.py`. Tests use small
   fake parts (`KeywordScorer`, `SentenceChunker`) so they need no API.
4. Log the run in `EXPERIMENTS.md`, and log the decision in
   `decisions.md` if it changes the architecture.

`DenseScorer` embeds units through the starter's cache
(`<index_dir>/embedding-cache.sqlite`). Cache keys include the model
settings and the exact text, so a new view or chunker embeds only new text,
and models never mix.

## Files We Added

| Path | Role |
| --- | --- |
| `src/qasper_agent/components.py` | `build_component`: builds a part from its config section, using the starter's `load_class` |
| `src/qasper_agent/retrieval/base.py` | `Unit` and the part interfaces (`Chunker`, `TextView`, `Scorer`, `Fusion`, `Reranker`) |
| `src/qasper_agent/retrieval/composed.py` | `ComposedRetriever` and `best_paragraphs` (rolls chunk scores up to paragraphs) |
| `src/qasper_agent/retrieval/chunkers.py` | Chunkers |
| `src/qasper_agent/retrieval/views.py` | Text views |
| `src/qasper_agent/retrieval/dense.py` | `DenseScorer` (embeddings, cosine similarity) |
| `src/qasper_agent/retrieval/fusion.py` | `ReciprocalRankFusion` |
| `config/composed-dense.json` | The baseline built from composed parts; reference for retrieval experiments |
| `tests/test_composed_retriever.py` | Tests for the parts, scope, `top_k`, chunk rollup, reranking, and parity with the starter retriever |
| `tools/error_analysis.py` | Tags each question in a scored run (`retrieval_miss`, `missed_abstain`, ...) and writes a markdown report. Reads gold, so use it only after prediction |
| `EXPERIMENTS.md` | Running log of commands, metrics, failure analysis and decisions |
| `experiments/runs/` | Per-question error reports and metrics for each logged run |
| `docs/decisions.md` | Architecture decision log |
| `docs/extensions.md` | This guide |

Starter files changed: only `pyproject.toml`, to install `qasper_agent`
([D4](decisions.md#d4-put-our-code-in-srcqasper_agent-next-to-the-starter)).

## Planned

- `tools/retrieval_eval.py`: recall@k for any retriever config, with no
  answer model calls.
- `retrieval/bm25.py` (keyword scorer) and a section-title view, the first
  experiments from EXPERIMENTS.md entry 001.

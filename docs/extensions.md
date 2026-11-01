# Guide to Our Additions

What we added on top of the starter, where it lives, and how the pieces fit
together. The starter itself is described in [baseline.md](baseline.md).
Design reasons are in [decisions.md](decisions.md), and results are in
[EXPERIMENTS.md](../EXPERIMENTS.md).

## Principles

- **The starter is edited only deliberately** ([D1](decisions.md#d1-extend-the-starter-through-its-plug-in-points-edit-it-only-deliberately)).
  Our code plugs into it through the `"class"` keys in config files. A
  starter edit (a general hook, or a change where a plug-in would only
  duplicate the starter, like the prompt) gets its own commit and decision
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
| Text view | `render(paragraph, piece) -> str` | `views:RawText`, `views:SectionPrefixed` | `view` (default `RawText`) |
| Scorer | `fit(units)`, `score(query, candidates) -> list[float]`; exactly 0 means no match | `dense:DenseScorer`, `bm25:BM25Scorer` | `scorers` (list, required) |
| Fusion | `fuse(score_lists) -> list[float]` | `fusion:ReciprocalRankFusion` (no credit for a score of 0; equal scores share a rank) | `fusion` (default RRF, `k` = 60) |
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

### Building an Index

Index once with the usual command; `ComposedRetriever.prepare_index` runs
through the starter's hook
([D5](decisions.md#d5-add-an-opt-in-prepare_index-hook-to-qasper-rag-index))
and fills every scorer's cache:

```bash
qasper-rag index --corpus data/corpus.jsonl \
  --config config/composed-dense.json --index-dir .index-composed
```

`predict`, `answer` and `serve` then load from the cache. If a config
changes the text view, chunker or embedding model, rerun the command on
the same `--index-dir`: only new text gets embedded.

## Files We Added

| Path | Role |
| --- | --- |
| `src/qasper_agent/components.py` | `build_component`: builds a part from its config section, using the starter's `load_class` |
| `src/qasper_agent/retrieval/base.py` | `Unit` and the part interfaces (`Chunker`, `TextView`, `Scorer`, `Fusion`, `Reranker`) |
| `src/qasper_agent/retrieval/composed.py` | `ComposedRetriever` and `best_paragraphs` (rolls chunk scores up to paragraphs) |
| `src/qasper_agent/retrieval/chunkers.py` | Chunkers |
| `src/qasper_agent/retrieval/views.py` | Text views |
| `src/qasper_agent/retrieval/dense.py` | `DenseScorer` (embeddings, cosine similarity). Paragraph and question embeddings both go through the starter's cache, so reruns use identical vectors ([D14](decisions.md#d14-cache-question-embeddings-in-densescorer)) |
| `src/qasper_agent/retrieval/fusion.py` | `ReciprocalRankFusion` |
| `config/composed-dense.json` | The baseline built from composed parts; reference for retrieval experiments |
| `config/composed-bm25.json` | BM25 only; no API calls |
| `config/composed-dense-section.json`, `composed-bm25-section.json`, `composed-hybrid-section.json` | The dense, BM25 and hybrid configs with the section-title view; each differs from its raw-text config only in `view` (D9) |
| `config/composed-hybrid.json` | BM25 + dense, reciprocal rank fusion. Differs from `composed-dense.json` only by the added BM25 scorer |
| `src/qasper_agent/retrieval/bm25.py` | `BM25Scorer` (Okapi BM25, standard library) and its `tokenize` (lowercase, alphanumeric tokens, plural "s" dropped) |
| `tests/test_bm25.py` | Tests for tokenizing, BM25 scoring, RRF without credit for non-matches, hybrid ranking, the paper filter, and top-k prefix consistency |
| `tests/test_composed_retriever.py` | Tests for the parts, scope, `top_k`, chunk rollup, reranking, index-time caching, and parity with the starter retriever |
| `data/eval-120/` | 120 dev questions (100 answerable, 20 unanswerable, 94 papers) and gold, made with `tools/prepare_data.py --count 120 --abstentions 20`. A superset of the shipped 20. Used for retrieval evaluation (D6). Gold is for scoring only |
| `tools/retrieval_eval.py` | Retrieval-only evaluation for any retriever config: recall@5/10/20 (best reference per depth, as `score.py`), each question's top-20 ranking, and the paragraphs of its best-matching reference (chosen at recall@5) that the top 5 missed, with rank and section. Reads gold only after retrieval. No answer-model calls |
| `tests/test_retrieval_eval.py` | Tests for recall, best-reference choice and ties, missed paragraphs, the command on fixtures, and top-k prefix consistency (`retrieve(q, 20)[:5] == retrieve(q, 5)`) for the composed and starter retrievers |
| `tools/error_analysis.py` | Tags each question in a scored run (`retrieval_miss`, `missed_abstain`, ...) and writes a markdown report. Reads gold, so use it only after prediction |
| `EXPERIMENTS.md` | Running log of commands, metrics, failure analysis and decisions |
| `experiments/runs/` | Per-question error reports and metrics for each logged run |
| `docs/decisions.md` | Architecture decision log |
| `docs/extensions.md` | This guide |

Starter files changed:

| File | Change | Decision |
| --- | --- | --- |
| `pyproject.toml` | Installs `qasper_agent` too | [D4](decisions.md#d4-put-our-code-in-srcqasper_agent-next-to-the-starter) |
| `src/qasper_rag/commands/index.py` | Opt-in `prepare_index` hook for custom retrievers | [D5](decisions.md#d5-add-an-opt-in-prepare_index-hook-to-qasper-rag-index) |
| `docs/baseline.md` | Documents the hook under "Extending the Starter" | D5 |
| `tests/test_index_hook.py` (new) | Tests the hook | D5 |
| `src/qasper_rag/pipeline/generation.py` | `PROMPT`: abstain instead of hedging; exactly Yes or No, short answers | [D12](decisions.md#d12-fix-abstention-and-answer-length-in-the-starter-prompt) |

## Planned

- Stopword removal for BM25 queries, the first candidate from
  EXPERIMENTS.md 003 (weak function-word matches cause most hybrid losses).


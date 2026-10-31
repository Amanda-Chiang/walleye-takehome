# Architecture Decisions

A log of the design choices made on top of the starter, newest at the bottom.
Each entry says what was decided, why, what else was considered, and what it
costs. Status is **Accepted**, **Proposed** (not built yet, open to change) or
**Superseded** (with a link to the entry that replaced it). Evidence for
decisions driven by results lives in [EXPERIMENTS.md](../EXPERIMENTS.md).

---

## D1: Extend the starter through its plug-in points; do not edit it

- **Status:** Accepted, 2026-10-31
- **Decision:** New behavior lives in new modules outside `src/qasper_rag/`
  and is turned on by naming classes in a config file
  (`"class": "module:ClassName"`). Starter files stay unchanged. If a change
  to the starter becomes necessary, it gets its own decision entry
  explaining why the plug-in points were not enough.
- **Why:** The starter already provides the indexing, the output contract,
  quote repair, the CLI, the web page and the scorer, and it has tests. Leaving
  it untouched means every experiment is a config diff against a known
  baseline, `predict`, `answer` and `serve` all run our code without changes,
  and the contract checks in `RAGService` still guard every output.
- **Alternatives:** Fork and edit `src/qasper_rag/` directly. That is faster
  for a single change, but it mixes our code with reference code and makes
  it harder to show exactly what changed.
- **Cost:** Some work happens inside a retriever that would more naturally
  happen at index time (see D3).

## D2: Track every run in a log, with per-question error reports

- **Status:** Accepted, 2026-10-31
- **Decision:** `EXPERIMENTS.md` records every command, metric and decision.
  `tools/error_analysis.py` writes a per-question report for each run into
  `experiments/runs/`, which git tracks, unlike `results/`.
- **Why:** The brief values reading real failures, grouping them by cause,
  and fixing the most common cause first. A durable per-question record lets
  each decision point to its evidence.
- **Cost:** Each run needs a manual write-up.

## D3: Make retrieval a set of small, swappable strategies

- **Status:** Accepted, 2026-10-31. Skeleton built in
  `src/qasper_agent/retrieval/`. With raw text, whole paragraphs and dense
  scoring, it returns the same top 5 as the baseline on all 20 dev questions
  (EXPERIMENTS.md, C11).
- **Decision:** Split retrieval into roles that can each be replaced from
  config without touching the others:
  - **Text view:** what text represents a paragraph when indexing it, e.g.
    the raw text, or the section title followed by the text.
  - **Chunker:** splits a paragraph into smaller indexed units that keep
    their parent `paragraph_id`, e.g. one unit per paragraph or overlapping
    sentence windows.
  - **Scorer:** ranks units for a question, e.g. BM25 (standard library, no
    API) or dense embeddings (reusing the starter's `EmbeddingClient` and
    its SQLite cache).
  - **Fusion:** combines several scorers' rankings, e.g. reciprocal rank
    fusion.
  - **Reranker (optional):** reorders the top candidates.

  One configurable retriever class puts these together. Scores on chunks
  are rolled up to their parent paragraph, so the retriever always returns
  original, unchanged paragraphs, as the output contract requires.
- **Why:** The starter makes whole stages swappable (retriever, generator,
  service), but nothing inside a stage. Hybrid search, section titles and
  chunking all touch retrieval, and each should be testable on its own.
- **Alternatives:** One retriever class per combination, e.g.
  `HybridSectionRetriever`. Simple at first, but the number of classes grows
  with every combination.
- **Cost:** `qasper-rag index` only embeds raw paragraph text. For other text
  views, the retriever embeds its own units when it first loads and caches
  them in `index_dir`. The first load of a new view costs API calls; later
  loads use the cache. Index loading does not count toward query latency.
- **Interface choices made while building it:**
  - Chunkers split *before* the text view renders, so a view such as
    "section title + text" applies to every chunk, not only the first.
  - Scorers receive candidate *positions* (indices into the fitted units),
    not unit objects. This keeps the per-paper filter in one place (the
    retriever), and scorers stay simple arrays.
  - A paragraph's score is the best score among its chunks (max).
  - Fusion works on scores aligned to the candidates, so rank-based (RRF)
    and score-based (weighted sum) fusion fit the same interface.
  - Each part receives its whole config section as `settings`, as the
    starter's stages do. `qasper_agent.components.build_component` wraps
    the starter's `load_class` for this.

## D4: Put our code in `src/qasper_agent/`, next to the starter

- **Status:** Accepted, 2026-10-31
- **Decision:** Our package lives at `src/qasper_agent/`, a sibling of the
  starter's `src/qasper_rag/`, with subpackages by role (`retrieval/` now;
  `generation/` and `service/` later), mirroring the starter's `pipeline/`
  split. Config files name our classes as
  `qasper_agent.retrieval.<module>:<Class>`.
- **Why:** Reading `src/` shows the two layers at once: the starter
  framework and our agent. Installing it like the starter means
  `qasper-rag` finds our classes from any folder, not only the project
  root.
- **Alternatives:** A root-level folder such as `my_rag/`, which the
  starter's docs suggest. That needs no packaging change, but it only
  loads when commands run from the project root, and it breaks the `src/`
  layout.
- **Cost:** One line changed in a starter file: `pyproject.toml`
  `include = ["qasper_rag*", "qasper_agent*"]`. Rerun
  `python -m pip install --editable .` after cloning.

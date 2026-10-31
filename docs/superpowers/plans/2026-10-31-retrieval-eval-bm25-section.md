# Retrieval Evaluation, BM25, Hybrid and Section Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure retrieval on its own over a larger question set, then use
that measurement to decide decisions D7 (reranker gate), D9 (section-title
context gate) and D10 (parent-child form), after adding a BM25 scorer, a
hybrid configuration and a section-title text view.

**Architecture:** Everything plugs into the existing `ComposedRetriever`
(`src/qasper_agent/retrieval/`): BM25 is a new `Scorer`, the section view is
a new `TextView`, and hybrid is a config file that lists two scorers (RRF
fusion already exists). A new `tools/retrieval_eval.py` builds any configured
retriever from an index folder, retrieves the top 20 per question, and only
then reads gold to compute recall@5/10/20 and a near-miss report. No starter
code changes.

**Tech Stack:** Python 3.10+ standard library only at runtime; `unittest`
for tests; `ruff` for format and lint; OpenRouter embeddings through the
starter's `EmbeddingClient` (dense configs only).

## Global Constraints

- Python 3.10 or later. Runtime code uses only the standard library.
- Type hints and NumPy-style docstrings on public functions. Keep functions
  short. Line length 80 (`ruff`, settings in `pyproject.toml`).
- Run `ruff format src/qasper_agent tools` and
  `ruff check src/qasper_agent tools` before each code commit.
- Tests use `unittest` (not pytest) and need no API key. Run the full suite
  with `python -m unittest discover -s tests` before each code commit.
- Never read `data/dev_gold.jsonl` or any gold labels during inference. The
  evaluation tool reads gold only after all retrieval has finished.
- Do not download or use the QASPER test split.
- Do not change `tools/score.py` or `src/qasper_rag/validation/contracts.py`.
- Never commit, print or log an API key. The key lives in `.env` (not
  exported). For commands that call the API, prefix with
  `source .venv/bin/activate && set -a && source .env && set +a && `.
  Never `cat` `.env`.
- API calls cost money. Only dense configs call the API (question
  embeddings, plus paragraph embeddings once per new text at index time).
- Commit at each checkpoint on branch `build-rag-agent` with a descriptive,
  multi-line message and no attribution trailer. Keep code commits separate
  from docs and log commits.
- After every command that runs tests, lint, indexing or evaluation, add a
  row to the Command Log table in `EXPERIMENTS.md` (next free number after
  `C14`): kind, exact command, API yes/no, result.
- Add every new file to the "Files We Added" table in `docs/extensions.md`.
- Out of scope: the stashed "combine scores per paragraph before fusion"
  change (`stash@{0}`), which only matters once paragraphs are split into
  chunks; rerankers; generator changes; the agentic loop; cross-paper
  questions.

## File Map

| File | Status | Responsibility |
| --- | --- | --- |
| `data/eval-120/dev_questions.jsonl`, `dev_gold.jsonl`, `manifest.json` | Create (generated) | 120 dev questions (100 answerable), a superset of the shipped 20 |
| `tools/retrieval_eval.py` | Create | Recall@k and near-miss report for any retriever config |
| `tests/test_retrieval_eval.py` | Create | Tests for recall, miss relations, rank buckets, and the CLI on fixtures |
| `src/qasper_agent/retrieval/bm25.py` | Create | `BM25Scorer` and `tokenize` |
| `tests/test_bm25.py` | Create | Tests for tokenizing, scoring and BM25 inside `ComposedRetriever` |
| `src/qasper_agent/retrieval/views.py` | Modify | Add `SectionPrefixed` view |
| `tests/test_composed_retriever.py` | Modify | Add tests for `SectionPrefixed` |
| `config/composed-bm25.json`, `composed-hybrid.json` | Create | BM25-only and BM25 + dense configs |
| `config/composed-dense-section.json`, `composed-bm25-section.json`, `composed-hybrid-section.json` | Create | Same three configs with the section view |
| `experiments/runs/002-retrieval-*.json` | Create (generated) | Retrieval results per config |
| `EXPERIMENTS.md`, `docs/decisions.md`, `docs/extensions.md` | Modify | Log, decisions and file guide |

---

### Task 1: Build the 120-question retrieval evaluation set (D6)

**Files:**
- Create: `data/eval-120/dev_questions.jsonl`, `data/eval-120/dev_gold.jsonl`, `data/eval-120/manifest.json`
- Modify: `EXPERIMENTS.md` (Command Log), `docs/extensions.md` (Files We Added)

**Interfaces:**
- Consumes: `tools/download_data.py`, `tools/prepare_data.py` (starter tools, unchanged).
- Produces: `data/eval-120/dev_questions.jsonl` (records `question_id`, `paper_id`, `question`) and `data/eval-120/dev_gold.jsonl` (records `question_id`, `references`), used by Task 5.

- [ ] **Step 1: Download the dev split**

Run: `source .venv/bin/activate && python tools/download_data.py --data-dir data --splits dev`
Expected: `data/qasper-dev-v0.3.json` exists. It is gitignored (`data/qasper-*.json`).

- [ ] **Step 2: Generate 120 questions, 20 unanswerable, into a separate folder**

Run: `python tools/prepare_data.py --source data/qasper-dev-v0.3.json --output-dir data/eval-120 --count 120 --abstentions 20`
Expected: prints a manifest with `"questions": 120`, `"unanswerable": 20`, and the same `source_sha256` as `data/manifest.json` (`2ae7ee62…af93`). Never pass `--output-dir data`: that would overwrite the shipped files.

- [ ] **Step 3: Check the corpus is identical and remove the duplicate**

Run: `cmp data/eval-120/corpus.jsonl data/corpus.jsonl && rm data/eval-120/corpus.jsonl`
Expected: no output from `cmp` (files identical), then the duplicate is removed.

- [ ] **Step 4: Check the shipped 20 questions are included**

Run:
```bash
python - <<'EOF'
import json
small = {json.loads(l)["question_id"] for l in open("data/dev_questions.jsonl")}
large = {json.loads(l)["question_id"] for l in open("data/eval-120/dev_questions.jsonl")}
print(f"shipped 20 included: {small <= large}, total {len(large)}")
EOF
```
Expected: `shipped 20 included: True, total 120`. Selection is by question-ID hash with per-answerability quotas, so the first 15 answerable and first 5 unanswerable are the shipped ones.

- [ ] **Step 5: Log and commit**

Add Command Log rows to `EXPERIMENTS.md` for Steps 1–4 (kind "Data", API "no"). Add this row to the "Files We Added" table in `docs/extensions.md`:

```markdown
| `data/eval-120/` | 120 dev questions (100 answerable, 20 unanswerable) and gold, made with `tools/prepare_data.py --count 120 --abstentions 20`. A superset of the shipped 20. Used for retrieval evaluation (D6). Gold is for scoring only |
```

```bash
git add data/eval-120 EXPERIMENTS.md docs/extensions.md
git commit -m "Add a 120-question dev set for retrieval evaluation

Generated with tools/prepare_data.py --count 120 --abstentions 20 into
data/eval-120 (100 answerable, 20 unanswerable). Selection is by question-ID
hash, so it is a superset of the shipped 20. The corpus it wrote was
byte-identical to data/corpus.jsonl and was removed. Gold is for scoring
only (D6)."
```

---

### Task 2: Retrieval evaluation tool with a near-miss report

**Files:**
- Create: `tools/retrieval_eval.py`
- Test: `tests/test_retrieval_eval.py`
- Modify: `docs/extensions.md` (Files We Added, Planned)

**Interfaces:**
- Consumes: `qasper_rag.config.load_config`, `qasper_rag.config.load_class`, `qasper_rag.pipeline.retrieval.EmbeddingRetriever`, `qasper_rag.storage.jsonl.read_jsonl`; any retriever with `retrieve(question: dict, top_k: int) -> list[dict]`.
- Produces:
  - `retrieve_all(config_path: str | Path, index_dir: str | Path, questions: list[dict], depth: int) -> dict[str, list[str]]`
  - `recall_at(ranked: list[str], evidence_sets: list[list[str]], k: int) -> float`
  - `miss_relation(missed: str, top: list[str], paragraphs: dict[str, dict]) -> str` returning `"adjacent"`, `"same_section"` or `"unrelated"`
  - `rank_bucket(rank: int | None) -> str` returning `"6-10"`, `"11-20"` or `"below_20"`
  - `evaluate(ranked: dict[str, list[str]], gold: list[dict], paragraphs: dict[str, dict]) -> dict`
  - CLI: `python tools/retrieval_eval.py --config C --index-dir D --questions Q --gold G --output O`. Output JSON keys: `config`, `answerable_questions`, `recall_at_5`, `recall_at_10`, `recall_at_20`, `missed_gold_paragraphs`, `miss_relation`, `miss_rank`, `questions`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_eval.py`:

```python
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from qasper_rag.commands.index import build_index
from tools.retrieval_eval import (
    evaluate,
    main,
    miss_relation,
    rank_bucket,
    recall_at,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def para(pid, section):
    return {"paragraph_id": pid, "paper_id": pid.split(":")[0], "section": section, "text": "x"}


PARAGRAPHS = {p["paragraph_id"]: p for p in (
    para("a:p00001", "Results"), para("a:p00002", "Results"),
    para("a:p00005", "Results"), para("a:p00009", "Method"),
    *(para(f"a:p000{n}", "Other") for n in (20, 21, 22, 23)),
)}


class RecallTests(unittest.TestCase):
    def test_recall_uses_the_best_reference_like_score_py(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at(ranked, [["a", "x"], ["c"]], 1), 0.5)
        self.assertEqual(recall_at(ranked, [["a", "x"], ["c"]], 3), 1.0)


class MissTests(unittest.TestCase):
    def test_adjacent_beats_same_section(self):
        self.assertEqual(miss_relation("a:p00002", ["a:p00001"], PARAGRAPHS), "adjacent")

    def test_same_section_when_not_adjacent(self):
        self.assertEqual(miss_relation("a:p00005", ["a:p00001"], PARAGRAPHS), "same_section")

    def test_unrelated_otherwise(self):
        self.assertEqual(miss_relation("a:p00009", ["a:p00001"], PARAGRAPHS), "unrelated")
        self.assertEqual(miss_relation("a:p00009", [], PARAGRAPHS), "unrelated")

    def test_rank_buckets(self):
        self.assertEqual(rank_bucket(6), "6-10")
        self.assertEqual(rank_bucket(20), "11-20")
        self.assertEqual(rank_bucket(None), "below_20")


class EvaluateTests(unittest.TestCase):
    def test_skips_unanswerable_and_reports_misses(self):
        ranked = {
            "q1": ["a:p00009", "a:p00020", "a:p00021", "a:p00022", "a:p00023", "a:p00005"],
            "q2": ["a:p00001"],
        }
        gold = [
            {"question_id": "q1", "references": [
                {"answer": "y", "abstain": False, "evidence_paragraph_ids": ["a:p00009", "a:p00005"]}]},
            {"question_id": "q2", "references": [
                {"answer": "Unanswerable", "abstain": True, "evidence_paragraph_ids": []}]},
        ]
        result = evaluate(ranked, gold, PARAGRAPHS)
        self.assertEqual(result["answerable_questions"], 1)
        self.assertEqual(result["recall_at_5"], 0.5)
        self.assertEqual(result["recall_at_10"], 1.0)
        self.assertEqual(result["missed_gold_paragraphs"], 1)
        self.assertEqual(result["miss_rank"], {"6-10": 1})
        self.assertEqual(result["miss_relation"], {"unrelated": 1})
        self.assertEqual(result["questions"][0]["missed"],
                         [{"paragraph_id": "a:p00005", "rank": 6, "relation": "unrelated"}])


class CommandTests(unittest.TestCase):
    def test_runs_on_fixtures_without_an_api(self):
        config = {"retriever": {"class": "qasper_agent.retrieval.composed:ComposedRetriever",
                                "scorers": [{"class": "test_composed_retriever:KeywordScorer"}]},
                  "generator": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            build_index(FIXTURES / "corpus.jsonl", tmp, path)
            output = Path(tmp) / "out.json"
            main(Namespace(config=str(path), index_dir=tmp,
                           questions=str(FIXTURES / "questions.jsonl"),
                           gold=str(FIXTURES / "gold.jsonl"), output=str(output)))
            result = json.loads(output.read_text())
        self.assertEqual(result["answerable_questions"], 1)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["config"], str(path))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest discover -s tests -p "test_retrieval_eval.py" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.retrieval_eval'`.

- [ ] **Step 3: Write the tool**

Create `tools/retrieval_eval.py`:

```python
r"""Measure retrieval alone: evidence recall at several depths, no answers.

Builds the retriever named in a config from an index folder and retrieves
the top 20 paragraphs for every question, searching only the question's
paper. Only then does it read the gold file, to compute recall@5, @10 and
@20 the way ``tools/score.py`` computes recall@5 (best reference per
question). For each gold paragraph missing from the top 5 it also reports
how far down it was ranked and how it relates to the top 5: ``adjacent``
(the paragraph just before or after one of them), ``same_section``, or
``unrelated``. Dense retrievers only call the API to embed the questions.

Example::

    python tools/retrieval_eval.py --config config/composed-bm25.json \
      --index-dir .index-eval120 --questions results/eval-120/questions.jsonl \
      --gold results/eval-120/gold.jsonl \
      --output experiments/runs/002-retrieval-composed-bm25.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.config import load_class, load_config
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.storage.jsonl import read_jsonl

Record = dict[str, Any]
DEPTHS = (5, 10, 20)
ANSWER_DEPTH = 5  # the most paragraphs the output contract allows


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    return parser


def retrieve_all(
    config_path: str | Path,
    index_dir: str | Path,
    questions: list[Record],
    depth: int,
) -> dict[str, list[str]]:
    """Return the top ``depth`` paragraph IDs for each question, best first.

    Parameters
    ----------
    config_path : str or Path
        The settings file naming the retriever.
    index_dir : str or Path
        The index folder built with that settings file.
    questions : list[dict]
        Records with ``question_id``, ``paper_id`` and ``question``.
    depth : int
        How many paragraphs to retrieve per question.

    Returns
    -------
    dict[str, list[str]]
        Paragraph IDs per question ID.
    """
    config = load_config(config_path)
    paragraphs = read_jsonl(Path(index_dir) / "paragraphs.jsonl")
    retriever_class = load_class(config.retriever, EmbeddingRetriever)
    retriever = retriever_class(paragraphs, config.retriever, index_dir)
    return {
        q["question_id"]: [
            p["paragraph_id"]
            for p in retriever.retrieve(
                {"question": q["question"], "paper_ids": [q["paper_id"]]},
                depth,
            )
        ]
        for q in questions
    }


def recall_at(ranked: list[str], evidence_sets: list[list[str]], k: int) -> float:
    """Return the best recall over references of the top ``k`` IDs."""
    top = set(ranked[:k])
    return max(len(top & set(e)) / len(e) for e in evidence_sets)


def paragraph_number(paragraph_id: str) -> int:
    """Return the position of a paragraph in its paper, e.g. 12 for p00012."""
    return int(paragraph_id.rsplit(":p", 1)[1])


def miss_relation(
    missed: str, top: list[str], paragraphs: dict[str, Record]
) -> str:
    """Say how a missed gold paragraph relates to the retrieved top ones.

    Parameters
    ----------
    missed : str
        A gold paragraph ID that is not in ``top``.
    top : list[str]
        The retrieved paragraph IDs the answer would use.
    paragraphs : dict[str, dict]
        Every indexed paragraph by ID.

    Returns
    -------
    str
        ``adjacent`` if it is just before or after a retrieved paragraph,
        else ``same_section`` if it shares a section with one, else
        ``unrelated``.
    """
    number = paragraph_number(missed)
    if any(abs(paragraph_number(pid) - number) == 1 for pid in top):
        return "adjacent"
    section = paragraphs[missed]["section"]
    if any(paragraphs[pid]["section"] == section for pid in top):
        return "same_section"
    return "unrelated"


def rank_bucket(rank: int | None) -> str:
    """Group the rank of a missed gold paragraph (``None`` if not found)."""
    if rank is None:
        return f"below_{DEPTHS[-1]}"
    return "6-10" if rank <= DEPTHS[1] else f"11-{DEPTHS[-1]}"


def evaluate(
    ranked: dict[str, list[str]],
    gold: list[Record],
    paragraphs: dict[str, Record],
) -> dict[str, Any]:
    """Score retrieval for the answerable gold questions.

    Recall uses the best reference per question, as ``tools/score.py`` does.
    The near-miss counts cover every gold paragraph of any reference that
    is missing from the top 5.

    Parameters
    ----------
    ranked : dict[str, list[str]]
        Retrieved paragraph IDs per question, best first.
    gold : list[dict]
        Gold records with ``question_id`` and ``references``.
    paragraphs : dict[str, dict]
        Every indexed paragraph by ID.

    Returns
    -------
    dict
        Summary metrics and a per-question breakdown.
    """
    recalls: dict[int, list[float]] = {k: [] for k in DEPTHS}
    relations: Counter[str] = Counter()
    ranks: Counter[str] = Counter()
    per_question = []
    for row in gold:
        evidence = [
            ref["evidence_paragraph_ids"]
            for ref in row["references"]
            if not ref["abstain"]
        ]
        if not evidence:
            continue
        ids = ranked[row["question_id"]]
        for k in DEPTHS:
            recalls[k].append(recall_at(ids, evidence, k))
        top = ids[:ANSWER_DEPTH]
        missed = []
        for pid in sorted({p for e in evidence for p in e} - set(top)):
            rank = ids.index(pid) + 1 if pid in ids else None
            relation = miss_relation(pid, top, paragraphs)
            relations[relation] += 1
            ranks[rank_bucket(rank)] += 1
            missed.append({"paragraph_id": pid, "rank": rank, "relation": relation})
        per_question.append(
            {
                "question_id": row["question_id"],
                "recall_at_5": recalls[ANSWER_DEPTH][-1],
                "missed": missed,
            }
        )
    count = len(recalls[ANSWER_DEPTH])
    return {
        "answerable_questions": count,
        **{f"recall_at_{k}": sum(v) / count if count else 0.0 for k, v in recalls.items()},
        "missed_gold_paragraphs": sum(relations.values()),
        "miss_relation": dict(relations),
        "miss_rank": dict(ranks),
        "questions": per_question,
    }


def main(args: argparse.Namespace) -> None:
    """Retrieve for every question, then score against gold."""
    questions = read_jsonl(args.questions)
    ranked = retrieve_all(args.config, args.index_dir, questions, DEPTHS[-1])
    # Gold is read only after every question has been retrieved.
    gold = read_jsonl(args.gold)
    paragraphs = {
        p["paragraph_id"]: p
        for p in read_jsonl(Path(args.index_dir) / "paragraphs.jsonl")
    }
    result = {"config": args.config, **evaluate(ranked, gold, paragraphs)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "questions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(setup_args().parse_args())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest discover -s tests -p "test_retrieval_eval.py" -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Format, lint, full suite**

Run: `ruff format tools/retrieval_eval.py && ruff check tools/retrieval_eval.py && python -m unittest discover -s tests`
Expected: lint clean (fix any line-length findings by wrapping), all tests OK.

- [ ] **Step 6: Update the guide, log and commit**

In `docs/extensions.md`, add to "Files We Added":

```markdown
| `tools/retrieval_eval.py` | Retrieval-only evaluation: recall@5/10/20 for any retriever config, and for each gold paragraph missed from the top 5, its rank (6–10, 11–20, below 20) and relation to the top 5 (adjacent, same section, unrelated). Reads gold only after retrieval. Feeds D7, D9 and D10 |
| `tests/test_retrieval_eval.py` | Tests for recall, miss relations, rank buckets, and the command on fixtures |
```

and remove the `tools/retrieval_eval.py` bullet from "Planned". Log the test and lint commands in `EXPERIMENTS.md`.

```bash
git add tools/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "Add retrieval-only evaluation with a near-miss report

tools/retrieval_eval.py builds the retriever named in a config, retrieves
the top 20 paragraphs per question within its paper, and only then reads
gold. It reports recall@5/10/20 (best reference per question, as score.py
does for recall@5) and, for each gold paragraph missed from the top 5, its
rank bucket and whether it is adjacent to, in the same section as, or
unrelated to the retrieved paragraphs. These answer the reranker gate (D7)
and the parent-child choice (D10). No answer-model calls."
git add docs/extensions.md EXPERIMENTS.md
git commit -m "Document the retrieval evaluation tool and log its checks"
```

---

### Task 3: BM25 scorer, and BM25-only and hybrid configs

**Files:**
- Create: `src/qasper_agent/retrieval/bm25.py`
- Create: `config/composed-bm25.json`, `config/composed-hybrid.json`
- Test: `tests/test_bm25.py`
- Modify: `docs/extensions.md` (part table, Files We Added)

**Interfaces:**
- Consumes: `qasper_agent.retrieval.base.Unit` (fields `paragraph_id`, `paper_id`, `text`); the `Scorer` protocol (`fit(units)`, `score(query, candidates) -> list[float]`); constructor `BM25Scorer(settings: dict, index_dir: str | Path)`.
- Produces: `qasper_agent.retrieval.bm25:BM25Scorer` with settings `k1` (default 1.5) and `b` (default 0.75); `tokenize(text: str) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bm25.py`:

```python
import unittest

from qasper_agent.retrieval.base import Unit
from qasper_agent.retrieval.bm25 import BM25Scorer, tokenize
from qasper_agent.retrieval.composed import ComposedRetriever


def fitted(*texts, **settings):
    scorer = BM25Scorer(settings, "unused")
    scorer.fit([Unit(f"p:p{i:05d}", "p", t) for i, t in enumerate(texts)])
    return scorer


class TokenizeTests(unittest.TestCase):
    def test_lowercases_splits_and_strips_plural_s(self):
        self.assertEqual(tokenize("Our Baselines: TransE, 82% F1."),
                         ["our", "baseline", "transe", "82", "f1"])

    def test_keeps_short_words_and_double_s(self):
        self.assertEqual(tokenize("is class gas"), ["is", "class", "gas"])


class ScoreTests(unittest.TestCase):
    def test_matching_unit_beats_non_matching(self):
        scores = fitted("our baseline is TransE", "we use embeddings").score(
            "What baselines are used?", [0, 1])
        self.assertGreater(scores[0], scores[1])

    def test_no_overlap_scores_zero(self):
        self.assertEqual(fitted("alpha beta").score("gamma", [0]), [0.0])

    def test_rare_term_outweighs_common_term(self):
        scorer = fitted("model data", "model data", "model transe", "model")
        rare, common = scorer.score("transe", [2]), scorer.score("model", [2])
        self.assertGreater(rare[0], common[0])

    def test_shorter_unit_wins_on_equal_term_count(self):
        scorer = fitted("baseline", "baseline " + "filler " * 30)
        short, long = scorer.score("baseline", [0, 1])
        self.assertGreater(short, long)

    def test_scores_only_requested_candidates(self):
        self.assertEqual(len(fitted("a b", "b c", "c d").score("b", [2, 0])), 2)


class RetrieverTests(unittest.TestCase):
    def test_bm25_inside_composed_retriever(self):
        paragraphs = [
            {"paragraph_id": "x:p00000", "paper_id": "x", "section": "Intro", "text": "We use word embeddings."},
            {"paragraph_id": "x:p00001", "paper_id": "x", "section": "Experiments", "text": "Our baseline is TransE."},
        ]
        settings = {"scorers": [{"class": "qasper_agent.retrieval.bm25:BM25Scorer"}]}
        retriever = ComposedRetriever(paragraphs, settings, "unused")
        passages = retriever.retrieve({"question": "What baselines are used?", "paper_ids": ["x"]}, 1)
        self.assertEqual(passages[0]["paragraph_id"], "x:p00001")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest discover -s tests -p "test_bm25.py" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qasper_agent.retrieval.bm25'`.

- [ ] **Step 3: Write the scorer**

Create `src/qasper_agent/retrieval/bm25.py`:

```python
"""Score units by BM25 keyword relevance, with no API calls."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qasper_agent.retrieval.base import Unit

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, and drop a plural ``s``.

    The plural rule lets "baselines" match "baseline". It skips words of 3
    letters or fewer and words ending in "ss" (e.g. "class").
    """
    return [
        t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
        for t in TOKEN.findall(text.lower())
    ]


class BM25Scorer:
    """Rank units by Okapi BM25 over their tokens.

    Term rarity (IDF) is computed over every indexed unit, all papers
    together, so words that are common across papers count for little.

    Parameters
    ----------
    settings : dict
        ``k1`` (default 1.5) limits how much repeated terms add. ``b``
        (default 0.75) sets how much long units are penalized.
    index_dir : str or Path
        Unused; BM25 needs no cache.
    """

    def __init__(self, settings: dict[str, Any], index_dir: str | Path) -> None:
        self.k1: float = settings.get("k1", 1.5)
        self.b: float = settings.get("b", 0.75)
        self.terms: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.average_length = 1.0
        self.idf: dict[str, float] = {}

    def fit(self, units: list[Unit]) -> None:
        """Count the terms of every unit and their rarity."""
        self.terms = [Counter(tokenize(unit.text)) for unit in units]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        frequency = Counter(term for terms in self.terms for term in terms)
        count = len(self.terms)
        self.idf = {
            term: math.log(1 + (count - df + 0.5) / (df + 0.5))
            for term, df in frequency.items()
        }

    def score(self, query: str, candidates: list[int]) -> list[float]:
        """Return the BM25 score of each candidate for ``query``."""
        query_terms = list(dict.fromkeys(tokenize(query)))
        return [self._score(query_terms, i) for i in candidates]

    def _score(self, query_terms: list[str], position: int) -> float:
        terms = self.terms[position]
        length = self.lengths[position] / self.average_length
        norm = self.k1 * (1 - self.b + self.b * length)
        return sum(
            self.idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm)
            for t in query_terms
            if terms[t]
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest discover -s tests -p "test_bm25.py" -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Add the configs**

Create `config/composed-bm25.json`:

```json
{
  "top_k": 5,
  "retriever": {
    "class": "qasper_agent.retrieval.composed:ComposedRetriever",
    "chunker": {"class": "qasper_agent.retrieval.chunkers:WholeParagraph"},
    "view": {"class": "qasper_agent.retrieval.views:RawText"},
    "scorers": [
      {"class": "qasper_agent.retrieval.bm25:BM25Scorer", "k1": 1.5, "b": 0.75}
    ],
    "fusion": {"class": "qasper_agent.retrieval.fusion:ReciprocalRankFusion", "k": 60}
  },
  "generator": {
    "temperature": 0,
    "max_tokens": 4096,
    "attempts": 3,
    "timeout_s": 60,
    "model": "openai/gpt-6-luna",
    "base_url": "https://openrouter.ai/api/v1",
    "reasoning": {"effort": "none"},
    "structured_output": true
  }
}
```

Create `config/composed-hybrid.json`:

```json
{
  "top_k": 5,
  "retriever": {
    "class": "qasper_agent.retrieval.composed:ComposedRetriever",
    "chunker": {"class": "qasper_agent.retrieval.chunkers:WholeParagraph"},
    "view": {"class": "qasper_agent.retrieval.views:RawText"},
    "scorers": [
      {"class": "qasper_agent.retrieval.bm25:BM25Scorer", "k1": 1.5, "b": 0.75},
      {
        "class": "qasper_agent.retrieval.dense:DenseScorer",
        "embedding": {
          "model": "openai/text-embedding-3-small",
          "dimensions": 256,
          "batch_size": 32,
          "max_input_chars": 12000,
          "attempts": 3,
          "timeout_s": 60
        }
      }
    ],
    "fusion": {"class": "qasper_agent.retrieval.fusion:ReciprocalRankFusion", "k": 60}
  },
  "generator": {
    "temperature": 0,
    "max_tokens": 4096,
    "attempts": 3,
    "timeout_s": 60,
    "model": "openai/gpt-6-luna",
    "base_url": "https://openrouter.ai/api/v1",
    "reasoning": {"effort": "none"},
    "structured_output": true
  }
}
```

Check both load: `python -c "from qasper_rag.config import load_config as l; [l(f'config/composed-{n}.json') for n in ('bm25','hybrid')]; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Format, lint, full suite, then commit code**

Run: `ruff format src/qasper_agent && ruff check src/qasper_agent && python -m unittest discover -s tests`
Expected: lint clean, all tests OK.

```bash
git add src/qasper_agent/retrieval/bm25.py tests/test_bm25.py config/composed-bm25.json config/composed-hybrid.json
git commit -m "Add a BM25 scorer and BM25-only and hybrid configs

BM25Scorer is a standard-library Okapi BM25 Scorer for ComposedRetriever
(k1=1.5, b=0.75, IDF over all indexed units). Its tokenizer lowercases,
splits on non-alphanumerics, and drops a plural s so \"baselines\" matches
\"baseline\" (a run 001 miss). config/composed-bm25.json uses it alone and
needs no API; config/composed-hybrid.json fuses it with the dense scorer by
reciprocal rank fusion, so each can be measured separately."
```

- [ ] **Step 7: Update the guide, log and commit**

In `docs/extensions.md`, change the Scorer row of the part table to list `dense:DenseScorer`, `bm25:BM25Scorer`, and add to "Files We Added":

```markdown
| `src/qasper_agent/retrieval/bm25.py` | `BM25Scorer` (keyword relevance, no API) and its `tokenize` |
| `tests/test_bm25.py` | Tests for tokenizing, BM25 scoring and BM25 inside the composed retriever |
| `config/composed-bm25.json` | BM25 only |
| `config/composed-hybrid.json` | BM25 + dense, reciprocal rank fusion |
```

Log the commands in `EXPERIMENTS.md`, then:

```bash
git add docs/extensions.md EXPERIMENTS.md
git commit -m "Document the BM25 scorer and hybrid configs and log checks"
```

---

### Task 4: Section-title text view and section configs (D9)

**Files:**
- Modify: `src/qasper_agent/retrieval/views.py`
- Create: `config/composed-dense-section.json`, `config/composed-bm25-section.json`, `config/composed-hybrid-section.json`
- Test: `tests/test_composed_retriever.py`
- Modify: `docs/extensions.md`

**Interfaces:**
- Consumes: the `TextView` protocol, `render(paragraph: dict, piece: str) -> str`; constructor `View(settings: dict)`; `BM25Scorer` from Task 3.
- Produces: `qasper_agent.retrieval.views:SectionPrefixed`, which renders `"<section path>\n<piece>"` with `" ::: "` replaced by `" > "`, and the piece unchanged when the section is empty.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_composed_retriever.py` (imports at the top, class before `if __name__`):

```python
from qasper_agent.retrieval.views import SectionPrefixed
```

```python
class SectionViewTests(unittest.TestCase):
    def test_prefixes_the_section_path(self):
        view = SectionPrefixed({})
        paragraph = {"section": "Results ::: Parsing", "text": "LAS is 80."}
        self.assertEqual(view.render(paragraph, "LAS is 80."), "Results > Parsing\nLAS is 80.")

    def test_empty_section_leaves_text_unchanged(self):
        self.assertEqual(SectionPrefixed({}).render({"section": "", "text": "t"}, "t"), "t")

    def test_section_words_become_searchable(self):
        # Run 001, Q13: the evidence never says "baseline"; its section does.
        paragraphs = [
            {"paragraph_id": "x:p00000", "paper_id": "x", "section": "Introduction",
             "text": "We study emotion detection and compare our approach to others."},
            {"paragraph_id": "x:p00001", "paper_id": "x", "section": "Baseline Approaches",
             "text": "We compare our approach to two other systems."},
        ]
        bm25 = [{"class": "qasper_agent.retrieval.bm25:BM25Scorer"}]
        question = {"question": "What baseline is used?", "paper_ids": ["x"]}
        raw = ComposedRetriever(paragraphs, {"scorers": bm25}, "unused")
        section = ComposedRetriever(
            paragraphs, {"scorers": bm25, "view": {"class": "qasper_agent.retrieval.views:SectionPrefixed"}},
            "unused")
        self.assertEqual(raw.retrieve(question, 1)[0]["paragraph_id"], "x:p00000")
        self.assertEqual(section.retrieve(question, 1)[0]["paragraph_id"], "x:p00001")
        self.assertIs(section.retrieve(question, 1)[0], paragraphs[1])
```

Note on the third test: with no overlapping words, both raw-text paragraphs score 0 and the tie keeps the earlier paragraph (`x:p00000`) first. With the section view, only `x:p00001` contains "baseline". The last assertion checks that the returned paragraph is the original dict, not the prefixed text.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest discover -s tests -p "test_composed_retriever.py" -v`
Expected: FAIL with `ImportError: cannot import name 'SectionPrefixed'`.

- [ ] **Step 3: Write the view**

Append to `src/qasper_agent/retrieval/views.py`:

```python
class SectionPrefixed:
    """Put the section path in front of the piece, e.g. "Results > Parsing".

    Some evidence paragraphs never use the question's words but sit under a
    section that does (run 001: "Baseline Approaches"). Only the indexed
    text changes; answers and quotes use the original paragraph.

    Parameters
    ----------
    settings : dict
        Unused.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def render(self, paragraph: Passage, piece: str) -> str:
        """Return the section path, a newline, then ``piece``.

        Returns ``piece`` unchanged when the paragraph has no section.
        """
        section = paragraph.get("section", "").replace(" ::: ", " > ")
        return f"{section}\n{piece}" if section else piece
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest discover -s tests -p "test_composed_retriever.py" -v`
Expected: all tests PASS.

- [ ] **Step 5: Add the three section configs**

Create each file as a copy of its raw-text counterpart with only the `view` line changed to `{"class": "qasper_agent.retrieval.views:SectionPrefixed"}`:

```bash
python - <<'EOF'
import json
for name in ("dense", "bm25", "hybrid"):
    config = json.load(open(f"config/composed-{name}.json"))
    config["retriever"]["view"] = {"class": "qasper_agent.retrieval.views:SectionPrefixed"}
    with open(f"config/composed-{name}-section.json", "w") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    print(f"config/composed-{name}-section.json")
EOF
git diff --no-index config/composed-hybrid.json config/composed-hybrid-section.json
```

Expected: three files written; the diff shows only the `view` class changed (plus JSON layout).

- [ ] **Step 6: Format, lint, full suite, then commit code**

Run: `ruff format src/qasper_agent && ruff check src/qasper_agent && python -m unittest discover -s tests`
Expected: lint clean, all tests OK.

```bash
git add src/qasper_agent/retrieval/views.py tests/test_composed_retriever.py config/composed-*-section.json
git commit -m "Add a section-title text view and section configs (D9)

SectionPrefixed puts the section path (\"Results > Parsing\") in front of
each paragraph's indexed text, so section words become searchable for BM25
and dense scoring. Answers and quotes still use the original paragraph.
Three configs pair it with dense, BM25 and hybrid scoring so the gate in D9
can be measured against the raw-text versions."
```

- [ ] **Step 7: Update the guide, log and commit**

In `docs/extensions.md`, change the Text view row of the part table to list `views:RawText`, `views:SectionPrefixed`, and add to "Files We Added":

```markdown
| `config/composed-dense-section.json`, `composed-bm25-section.json`, `composed-hybrid-section.json` | The dense, BM25 and hybrid configs with the section-title view (D9) |
```

Remove the "section-title view" bullet from "Planned". Log the commands, then:

```bash
git add docs/extensions.md EXPERIMENTS.md
git commit -m "Document the section-title view and log checks"
```

---

### Task 5: Run the retrieval comparison and record the decisions

**Files:**
- Create: `experiments/runs/002-retrieval-<config>.json` (6 files, generated)
- Modify: `EXPERIMENTS.md` (entry 002), `docs/decisions.md` (D7, D9, D10)

**Interfaces:**
- Consumes: `tools/retrieval_eval.py` (Task 2), the six `config/composed-*.json` files (Tasks 3–4), `data/eval-120/` (Task 1), `tools/subset.py`, `qasper-rag index`.
- Produces: entry 002 in `EXPERIMENTS.md` and updated statuses for D7, D9 and D10.

- [ ] **Step 1: Write the decision rules down before running**

Add these lines to the "Evidence to collect" or "Decision" part of each entry in `docs/decisions.md`, then commit (`git commit -m "Set decision thresholds for D7, D9 and D10 before measuring"`). Writing them first keeps the results from shaping the rules. They are proposals; Amanda can change them before this step runs.

- D7: build a reranker if, for the best config, `recall_at_20 − recall_at_5 ≥ 0.10` (10 of 100 answerable questions' worth of evidence ranked 6–20).
- D9: keep the section view for a scorer if its `recall_at_5` beats the raw-text version by `≥ 0.03`, and list the questions it wins and loses.
- D10: choose the parent-child form by the largest `miss_relation` category for the best config: `same_section` → section filling; `adjacent` → neighbours; `unrelated` → none (work on scoring).

- [ ] **Step 2: Build the subset corpus**

Run: `python tools/subset.py --corpus data/corpus.jsonl --questions data/eval-120/dev_questions.jsonl --gold data/eval-120/dev_gold.jsonl --count 120 --output-dir results/eval-120`
Expected: prints `120 questions (20 unanswerable), N papers, M paragraphs -> results/eval-120`. Record N and M.

- [ ] **Step 3: Index once per config into one shared folder**

All six configs share `.index-eval120`: `paragraphs.jsonl` is the same, and the embedding cache is keyed by model settings and text, so raw-text vectors are reused across dense configs and only the section view adds new text.

```bash
source .venv/bin/activate && set -a && source .env && set +a && \
for c in composed-bm25 composed-dense composed-hybrid composed-bm25-section composed-dense-section composed-hybrid-section; do
  qasper-rag index --corpus results/eval-120/corpus.jsonl --config config/$c.json --index-dir .index-eval120 || break
done
```

Expected: each run logs `Preparing ComposedRetriever for the index`. Only `composed-dense` and `composed-dense-section` embed new text (about M paragraphs each); the others finish quickly from cache or without the API.

- [ ] **Step 4: Run the retrieval evaluation for each config**

```bash
source .venv/bin/activate && set -a && source .env && set +a && \
for c in composed-bm25 composed-dense composed-hybrid composed-bm25-section composed-dense-section composed-hybrid-section; do
  python tools/retrieval_eval.py --config config/$c.json --index-dir .index-eval120 \
    --questions results/eval-120/questions.jsonl --gold results/eval-120/gold.jsonl \
    --output experiments/runs/002-retrieval-$c.json || break
done
```

Expected: six JSON files in `experiments/runs/`, each with `"answerable_questions": 100`.

- [ ] **Step 5: Print the comparison table**

```bash
python - <<'EOF'
import json
names = ["composed-bm25", "composed-dense", "composed-hybrid",
         "composed-bm25-section", "composed-dense-section", "composed-hybrid-section"]
print("| Config | R@5 | R@10 | R@20 | Missed | Adjacent | Same section | Unrelated | Rank 6-10 | 11-20 | Below 20 |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for n in names:
    r = json.load(open(f"experiments/runs/002-retrieval-{n}.json"))
    rel, rank = r["miss_relation"], r["miss_rank"]
    print(f"| {n} | {r['recall_at_5']:.2f} | {r['recall_at_10']:.2f} | {r['recall_at_20']:.2f} | "
          f"{r['missed_gold_paragraphs']} | {rel.get('adjacent', 0)} | {rel.get('same_section', 0)} | "
          f"{rel.get('unrelated', 0)} | {rank.get('6-10', 0)} | {rank.get('11-20', 0)} | {rank.get('below_20', 0)} |")
EOF
```

Expected: a markdown table with six rows.

Sanity check: on the 15 shipped answerable questions, `composed-dense` must
reproduce run 001's recall@5 (0.33), because the same retriever on the same
questions gives identical rankings (C11):

```bash
python -c "
import json
shipped = {json.loads(l)['question_id'] for l in open('data/dev_questions.jsonl')}
rows = json.load(open('experiments/runs/002-retrieval-composed-dense.json'))['questions']
picked = [q['recall_at_5'] for q in rows if q['question_id'] in shipped]
print(len(picked), round(sum(picked) / len(picked), 2))
"
```

Expected: `15 0.33`. If not, stop and investigate before comparing configs.

- [ ] **Step 6: Read the misses before concluding**

For the best config, open its JSON and read at least 10 `missed` entries next to the paragraph text (`grep '"<paragraph_id>"' results/eval-120/corpus.jsonl`). Note any pattern the counts hide (e.g. misses that are captions, very short paragraphs, or gold labels that look wrong). Numbers come second to reading failures (ASSIGNMENT.md, Evaluation).

- [ ] **Step 7: Write entry 002 in `EXPERIMENTS.md`**

Use the same layout as entry 001: date, commit, configs, data (`data/eval-120`, 100 answerable questions, N papers, M paragraphs), commands (Command Log numbers), the table from Step 5, the patterns from Step 6, then "Decision and why", applying the three rules from Step 1 exactly as written. Add Command Log rows for Steps 2–4 (dense configs: API yes, question and paragraph embeddings; BM25: API no).

- [ ] **Step 8: Update decision statuses**

In `docs/decisions.md`, update the status line of D7, D9 and D10 with the outcome and a pointer to entry 002, e.g. "Gate measured 2026-11-01 (EXPERIMENTS.md 002): R@20 − R@5 = 0.xx, so …". If the hybrid config wins, add a short D14 entry: "Use hybrid BM25 + dense as the default retriever", with the numbers.

- [ ] **Step 9: Commit**

```bash
git add experiments/runs/002-retrieval-*.json EXPERIMENTS.md docs/decisions.md
git commit -m "Log retrieval comparison 002 and settle D7, D9 and D10

Compares BM25, dense and hybrid retrieval, each with raw text and with the
section-title view, on 100 answerable dev questions: recall@5/10/20 and a
near-miss report. Applies the thresholds set before the run to the reranker
gate (D7), the section-context gate (D9) and the parent-child choice (D10)."
```

Stop here and report the table and decisions to Amanda before starting any follow-up work.

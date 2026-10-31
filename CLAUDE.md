# Guide for Coding Agents

This repository is a take-home assignment. A candidate builds a question
answering system over QASPER research papers. ASSIGNMENT.md is the brief,
and docs/ has the data, evaluation and baseline reference pages. Read
ASSIGNMENT.md before making changes. This file is the short version for an agent
working with the candidate.

## Rules

- The candidate owns the design. Ask before large rewrites, and explain the
  choices you make so the candidate can defend them.
- Never commit, print or log an API key. Keys come from `OPENROUTER_API_KEY`
  or `RAG_API_KEY` in the environment. `.env` files are not loaded.
- Never read `data/dev_gold.jsonl` or any gold labels during inference. Gold
  is only for scoring.
- Do not download or use the QASPER test split. It is held out.
- Do not change `tools/score.py` or `src/qasper_rag/validation/contracts.py`
  to make results look better. Propose metric changes separately.
- API calls cost money. Use the tiny corpus in `tests/fixtures/` or a subset before a full run.

## The Baseline Is Optional

The code in `src/qasper_rag/` is a reference. The candidate can build on it,
change it, or start from scratch. Their own evaluation scripts are also fine.
If they start from scratch, predictions should still follow the output format
in docs/baseline.md, so `tools/score.py` can score them.

## Setup and Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --editable .            # standard library only
python -m pip install --editable ".[ragas]"   # optional RAGAS scoring
python -m unittest discover -s tests          # tests need no API key
```

| Task | Command |
| --- | --- |
| Build an index | `qasper-rag index --corpus data/corpus.jsonl --config config/baseline.json --index-dir .index` |
| Answer one question | `qasper-rag answer --index-dir .index --question "..." [--paper-id ID]` (`--config` defaults to `config/baseline.json`, as for `predict` and `serve`) |
| Batch predictions | `qasper-rag predict --index-dir .index --questions data/dev_questions.jsonl --config config/baseline.json --output results/run.jsonl` |
| Score | `python tools/score.py --corpus ... --questions ... --gold ... --predictions ... --output ...` |
| Small evaluation | `python tools/run_eval.py --count 3` (builds its own small index) |
| Chat page | `qasper-rag serve --index-dir .index` |

The justfile has shortcuts (`just test`, `just eval run`, `just app start`).
`just eval run` and `just app start` build an index of the full corpus first,
which embeds 15,757 paragraphs the first time. `just test` needs no API key.

## Code Layout

```text
src/qasper_rag/
  pipeline/     service.py (RAGService), retrieval.py, generation.py
  providers/    api_client.py, embeddings.py
  validation/   contracts.py (output rules), quotes.py
  storage/      jsonl.py
  commands/     index, predict, answer
  web/          server.py, static/index.html
  cli.py, config.py
tools/          score.py, run_eval.py, subset.py, ragas_eval.py, data tools
tests/          unittest suite with a fake provider; tiny corpus in fixtures/
```

## Extending the Pipeline

Add a retriever, generator or service as a new class and name it in a config
file with `"class": "module:ClassName"`. Other keys in that section reach the
class as `settings`. Modules are found from the current folder first, so a
`my_rag/` folder in the project root works. You do not need to edit
`service.py`.

- Retriever: `__init__(paragraphs, settings, index_dir)` and
  `retrieve(question, top_k)`. Search only `question["paper_ids"]`, return at
  most `top_k` passages, and do not change their text.
- Generator: `__init__(settings)` and `generate(question, passages)`,
  returning `answer`, `abstain` and `citations`.

- Service: subclass `RAGService` and override `run(question)`, calling
  `self.retrieve` and `self.generate`. Use this for steps that span both
  stages, e.g. retrying retrieval. The base class still checks the output.

`tests/test_extension.py` has a working example of each.

## Output Rules

Every prediction is checked by `validation/contracts.py`:

- At most 5 retrieved paragraph IDs, all from the question's paper.
- Each citation quote is an exact substring of a retrieved paragraph.
- An abstention has the answer `Unanswerable` and no citations.
- An answer that is not an abstention has at least one citation.

## Code Style

- Python 3.10 or later. Runtime code uses only the standard library.
- Type hints and NumPy-style docstrings on public functions. Keep functions
  short.
- Run `ruff format src tools` and `ruff check src tools` (settings in
  `pyproject.toml`).
- Add or update tests for behavior you change, and run the suite before
  handing work back.

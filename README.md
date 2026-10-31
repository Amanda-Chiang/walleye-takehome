# Special Projects Take-Home Assignment: Build a RAG Agent for QASPER

This repository is the starting point for a take-home assignment. You build a
RAG agent that answers questions about 281 NLP research papers, with
citations, and says so when the papers do not have the answer.

**Start with [ASSIGNMENT.md](ASSIGNMENT.md).** It describes the task, the
rules, how we evaluate, and what to submit.

## What Is in This Repository

| Part | Where | Do you need it? |
| --- | --- | --- |
| The papers | `data/corpus.jsonl` | Yes. This is what your agent searches and answers from. [docs/data.md](docs/data.md) describes it. |
| Evaluation questions, gold answers and tools | `data/dev_questions.jsonl`, `data/dev_gold.jsonl`, `tools/` | Optional. Use them to measure your agent, or write your own evaluation. [docs/evaluation.md](docs/evaluation.md) explains how. |
| A baseline RAG agent | `src/qasper_rag/`, `config/`, `tests/` | Optional. A working reference you can build on, change or ignore. [docs/baseline.md](docs/baseline.md) explains how to run and extend it. |

`CLAUDE.md` is a short guide for coding agents such as Claude Code. It
repeats the rules and the main commands.

## Quickstart with the Baseline

You need Python 3.10 or later and an [OpenRouter](https://openrouter.ai) API
key. The baseline uses only the standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
python -m unittest discover -s tests        # no API key needed

export OPENROUTER_API_KEY='<your key>'
python tools/run_eval.py --count 3          # index only the papers of the first 3 questions, answer them, and score
```

To ask your own questions, build the full index once and start the chat page:

```bash
qasper-rag index --corpus data/corpus.jsonl --config config/baseline.json --index-dir .index
qasper-rag serve --index-dir .index --config config/baseline.json
```

Then open `http://127.0.0.1:8000`. The full index embeds all 15,757
paragraphs, which takes a few minutes. Windows, uv and `just` instructions
are in [docs/baseline.md](docs/baseline.md).

## Repository Layout

```text
ASSIGNMENT.md     the task
CLAUDE.md         guide for coding agents
docs/             data, evaluation and baseline reference pages
data/             the papers, plus questions and gold answers for evaluation
src/qasper_rag/   the baseline agent
config/           baseline settings
tools/            scoring, RAGAS, subset, and data download tools
tests/            unit tests, and a tiny corpus in tests/fixtures/
```

The papers come from [QASPER](https://allenai.org/data/qasper) v0.3 by the
Allen Institute for AI, shared under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

# Experiment Log

A running record of each evaluated configuration: what ran, what it scored,
why it failed, and what I decided to change next. Newest entries go at the
bottom. Every entry links to a per-question error report in
`experiments/runs/`, generated with:

```bash
python tools/error_analysis.py --run-dir results/<run> \
  --output experiments/runs/<NNN>-<name>.md
cp results/<run>/metrics.json experiments/runs/<NNN>-<name>.metrics.json
```

`results/` is gitignored, so the reports and metrics copied into
`experiments/runs/` are the durable record.

**Outcome tags** (assigned mechanically by `tools/error_analysis.py`):

| Tag | Meaning |
| --- | --- |
| `retrieval_miss` | Answerable, but no gold evidence paragraph in the top 5 |
| `retrieval_partial` | Answerable, some but not all gold evidence retrieved, answered |
| `retrieved_answered` | Answerable, all gold evidence retrieved, answered (answer may still be wrong) |
| `over_abstain` | Answerable, gold evidence retrieved, but the agent abstained |
| `missed_abstain` | Unanswerable, but the agent answered |
| `correct_abstain` | Unanswerable, and the agent abstained |

## Command Log

Every command that ran tests, built an index or produced predictions, in
order. "API" marks commands that call the model provider and cost money.
Commands that need the key are run with `.env` loaded into the shell first
(`set -a && source .env && set +a`); the starter does not read `.env`
itself.

| # | Date | Kind | Command | API | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | 2026-10-31 | Index (smoke) | `qasper-rag index --corpus tests/fixtures/corpus.jsonl ... --index-dir .index-smoke` (assumed from the docs) | yes | `.index-smoke/` is empty, so this run did not finish. Not investigated |
| C2 | 2026-10-31 | Eval, 10 questions | Worked example in `docs/evaluation.md`: `tools/subset.py --count 10 --output-dir results/ragas-10`, index into `.index-ragas10`, predict, score | yes | See entry 000 |
| C3 | 2026-10-31 | Eval, 20 questions | `python tools/run_eval.py --count 20 --config config/baseline.json` | yes | See entry 001. Embedded 1,226 paragraphs, 20 predictions |
| C4 | 2026-10-31 | Error analysis | `python tools/error_analysis.py --run-dir results/baseline-20 --output experiments/runs/001-baseline-20.md` | no | Outcome counts match the manual reading in entry 001 |
| C5 | 2026-10-31 | Unit tests | `python -m unittest discover -s tests` | no | 36 tests, OK (2.3 s). Unmodified starter plus `tools/error_analysis.py`, which has no tests yet |
| C6 | 2026-10-31 | Lint | `ruff format tools/error_analysis.py && ruff check tools/error_analysis.py` | no | Not run: `ruff` is not installed in `.venv`. Done later as C10 |
| C7 | 2026-10-31 | Setup | `python -m pip install --editable .` after adding `qasper_agent*` to `pyproject.toml` (D4) | no | `qasper_agent` importable |
| C8 | 2026-10-31 | Setup | `python -m pip install ruff` | no | Lint tool installed in `.venv` (dev only, not a runtime dependency) |
| C9 | 2026-10-31 | Unit tests | `python -m unittest discover -s tests` | no | 46 tests, OK: 36 starter tests + 10 in `tests/test_composed_retriever.py` |
| C10 | 2026-10-31 | Lint | `ruff format src/qasper_agent tools/error_analysis.py && ruff check src/qasper_agent tools/error_analysis.py` | no | Clean after fixing 3 issues in `tools/error_analysis.py`. Its report for 001 was regenerated and is byte-identical |
| C11 | 2026-10-31 | Parity check | Inline script: `ComposedRetriever` with `config/composed-dense.json` on `.index-baseline-20`, top 5 for the 20 questions, compared with `results/baseline-20/predictions.jsonl` | yes (20 question embeddings; paragraph vectors from cache) | **20/20 identical** retrieved IDs and order. The refactor preserves baseline retrieval. Load took 0.1 s |

---

## 000: Baseline, 10 dev questions (before this log started)

- **Date:** 2026-10-31
- **Config:** `config/baseline.json` (same as 001)
- **Data:** first 10 dev questions (9 answerable, 1 unanswerable)
- **Commands:** C2
- **Outputs:** `results/ragas-10/` (not copied into `experiments/runs/`; no
  RAGAS output was found in the folder)

| Metric | Value |
| --- | ---: |
| answer_token_f1 | 0.13 |
| paragraph_evidence_recall_at_5 | 0.28 |
| citation_evidence_f1 | 0.27 |
| abstention_recall | 0.00 |
| answerable_question_coverage | 0.67 |
| mean_query_latency_ms | 2,170 |

**Run-to-run variance (checked against 001 on the same 10 questions).** The
retrieved paragraphs were identical for all 10, and every abstain decision
was the same. The answer text differed on 6 of 10 even at temperature 0, so
small changes in token F1 between runs can be noise from the generator
alone. Retrieval metrics are deterministic for a fixed index. This first
check only confirmed that the full pipeline works; entry 001 replaces it.

---

## 001: Baseline, 20 dev questions

- **Date:** 2026-10-31
- **Commit:** `37f6e8b` (unmodified starter)
- **Config:** `config/baseline.json`. `openai/text-embedding-3-small` at 256
  dimensions, cosine top 5 within the question's paper, one
  `openai/gpt-6-luna` call (temperature 0, no reasoning, JSON schema output).
- **Data:** first 20 questions of `data/dev_questions.jsonl` (15 answerable,
  5 unanswerable), 18 papers, 1,226 paragraphs.
- **Commands:** C3 (run), C4 (error report), C5 (unit tests)
- **Report:** [experiments/runs/001-baseline-20.md](experiments/runs/001-baseline-20.md)

### Metrics

| Metric | Value |
| --- | ---: |
| answer_token_f1 | 0.22 |
| paragraph_evidence_recall_at_5 | 0.33 |
| citation_evidence_f1 | 0.27 |
| abstention_precision | 0.20 |
| abstention_recall | 0.20 |
| answerable_question_coverage | 0.73 |
| mean_query_latency_ms | 2,031 |

### Failures by cause

| Cause | Count | Questions |
| --- | ---: | --- |
| Retrieval misses all gold evidence | 9 / 15 answerable | Q0, Q2, Q3, Q5, Q8, Q9, Q11, Q12, Q13 |
| Answers instead of abstaining | 4 / 5 unanswerable | Q6, Q17, Q18, Q19 |
| Verbose answers lower token F1 | about 8 answered | e.g. Q4, Q7, Q11, Q14, Q15 |
| Retrieval finds only part of the evidence | 2 / 15 | Q1, Q15 |
| Abstains despite retrieving the evidence | 1 / 15 | Q16 |

**Retrieval misses (9).** Reading the missed paragraphs shows three
sub-patterns:

- *Short, generic questions embed poorly.* For "What is the baseline?" (Q0)
  the evidence literally says "as baseline models"; for Q8 it says "Our
  baseline is TransE". Neither reached the top 5. Exact keyword overlap would
  rank both first.
- *Section titles are not used.* Q13's evidence sits under "Baseline
  Approaches", Q12's under "Features.", and Q9's under a section that opens
  with "the state of the art on conversational systems". Only paragraph text
  is embedded, so this signal is lost.
- *Evidence spans adjacent paragraphs.* Q9's gold is five consecutive
  paragraphs (p00017 to p00021), one chatbot each.
- The rest are harder: Q2 needs inference (the answer is "no data, a
  pretrained model is used"), Q3 needs comparing numbers across languages,
  and Q5 depends on a figure.

**Answers instead of abstaining (4).** The model often knows the evidence is
missing but answers anyway. Its answers contain phrases like "the passages
do not establish…" (Q6), "does not provide…" (Q18) and "the passage does not
name them" (Q19). Q6 answers "No" while admitting the passages are silent.

**Wrong abstentions are mostly a retrieval problem.** The agent abstained
on 4 answerable questions. For 3 of them (Q2, Q3, Q8) no gold evidence was
retrieved, so abstaining was the right call given what the model saw. Only
Q16 is a generator error.

**Verbosity.** The prompt tells the model to name the paper_id of each
claim. That suits questions across papers, but in single-paper evaluation it
adds "(paper 1907.09369)" and explanations ("Yes. Paper 1705.01265 reports…"
where the gold is "Yes"), which lowers token F1 even when the answer is right.

**Caveats.** With 15 answerable questions, one question moves evidence
recall by about 7 points. Some gold labels are debatable: Q11's prediction
("Accuracy, LAS, UAS") largely matches the gold answers but cites different
paragraphs, so it scores low.

### Decision and why

1. **Fix retrieval first.** It is the largest cause (9 of 15 answerable). It
   also causes 3 of the 4 wrong abstentions, so better generation cannot help
   until the evidence reaches the model. It can be measured without calling
   the answer model (recall@k needs only retrieval), so trying variants costs
   almost nothing. Planned changes:
   - a retrieval-only evaluation script (recall@k for k = 5, 10, 20), to keep
     retrieval metrics separate from answer metrics;
   - hybrid retrieval: BM25 combined with embeddings (e.g. reciprocal rank
     fusion), aimed at the short, generic questions;
   - add section titles to the text that gets indexed or embedded;
   - try adding neighbouring paragraphs for evidence spread across a section.
2. **Then generation.** Tell the model to abstain whenever it would have to
   hedge (aimed at 4 missed abstentions plus Q16), and give short answers in
   single-paper mode (aimed at verbosity). These are prompt changes that are
   cheap to test once retrieval is stable.
3. **Get more questions** with `tools/prepare_data.py --count` before
   trusting small differences.

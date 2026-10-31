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
| C12 | 2026-10-31 | Unit tests | `python -m unittest discover -s tests` after the `prepare_index` hook (D5) | no | 48 tests, OK (2 new in `tests/test_index_hook.py`) |
| C13 | 2026-10-31 | Unit tests + lint | `python -m unittest discover -s tests`; `ruff check src/qasper_agent src/qasper_rag/commands/index.py` | no | 49 tests, OK (1 new: indexing fills the cache so loading embeds nothing). Lint clean |
| C14 | 2026-10-31 | Index + parity check | `qasper-rag index --corpus results/baseline-20/corpus.jsonl --config config/composed-dense.json --index-dir .index-composed-dense-20`, then load `ComposedRetriever` with `EmbeddingClient.embed` counted and compare top 5 with `results/baseline-20/predictions.jsonl` | yes (1,226 paragraph embeddings + 20 question embeddings) | Index 21 s. Load **0.07 s, 0 embedding calls**. Retrieval **20/20 identical** to baseline |

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
alone. Retrieval metrics are deterministic for a fixed index (corrected in
002: nearly, not exactly, for dense retrieval). This first
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

---

## 002: Dense retrieval baseline, 120 dev questions (retrieval only)

- **Date:** 2026-10-31
- **Commit:** `20a15c5` (evaluator), config unchanged since `d6f4623`
- **Config:** `config/composed-dense.json`: raw paragraph text, whole
  paragraphs, `text-embedding-3-small` at 256 dimensions, cosine, RRF
  over one scorer (keeps its order). Same rankings as the starter's
  `EmbeddingRetriever` (C11, C14).
- **Data:** `data/eval-120` (100 answerable questions, 94 papers); full
  corpus indexed (15,757 paragraphs), each question searching only its paper.
- **Commands:**
  - `qasper-rag index --corpus data/corpus.jsonl --config config/composed-dense.json --index-dir .index-full` (4 min 10 s; about $0.02)
  - `python tools/retrieval_eval.py --config config/composed-dense.json --index-dir .index-full --questions data/eval-120/dev_questions.jsonl --gold data/eval-120/dev_gold.jsonl --output experiments/runs/002-dense.json`
- **Results:** [experiments/runs/002-dense.json](experiments/runs/002-dense.json)

### Metrics

| Metric | Value |
| --- | ---: |
| recall@5 | 0.59 |
| recall@10 | 0.79 |
| recall@20 | 0.91 |
| Questions with recall@5 = 1 / partial / 0 | 54 / 12 / 34 |
| Missed paragraphs (best reference), ranked 6–10 / 11–20 / below 20 | 25 / 17 / 26 (68 total) |

### Checks

- **Shared index isolation:** `.index-full` holds only `paragraphs.jsonl`
  (the same 15,757 records as `data/corpus.jsonl`) and
  `embedding-cache.sqlite`.
- **Reproduces run 001:** for the shipped 20 questions, the top 5 matches
  `results/baseline-20/predictions.jsonl` on 20 of 20. Recall@5 on the 15
  shipped answerable questions is 0.33, as in 001.
- **Top-k prefix consistency on real data:**
  `retrieve(q, 5) == retrieve(q, 20)[:5]` on 119 of 120 questions in the
  first pass. A rerun of that question matched. Embedding all 120 questions
  twice gave bit-identical vectors for only 70 of 120, with component
  differences up to 5.4e-3, though no top 5 changed in that pair. **Dense
  retrieval is nearly, not exactly, repeatable**: the API returns slightly
  different question embeddings between calls, and about 1 in 120
  questions can swap a paragraph at the top-5 boundary. The retriever logic
  is prefix-consistent (unit tests); the variation comes from the
  embedding API. This corrects the claim in 000 that retrieval is
  deterministic. Differences of one question (0.01 recall) between runs
  are within this noise.

### What the numbers say

- **The shipped 20 are harder than average for dense retrieval.** Recall@5
  is 0.33 on their 15 answerable questions and 0.59 on all 100. A
  conclusion drawn from the 20 alone would have overstated the retrieval
  problem, which is why D6 exists.
- **There is headroom below rank 5.** 42 of 68 missed paragraphs are in
  the top 20 (recall@20 is 0.91). Per D7 this is a reason to consider
  ranking improvements later, not a decision. 26 missed paragraphs are not
  in the top 20 at all; reranking cannot reach those.

### Failures read (14 questions with recall@5 < 1)

| Pattern | Questions | Example |
| --- | ---: | --- |
| **List items under a section:** the question asks for a set; the evidence is several sibling paragraphs (one item each) that do not repeat the question's words, while a sentence *announcing* the list ranks first | 4 | "What is the state of the art described in the paper?": top 1 is "In the next section we present the state of the art…"; the gold paragraphs (ELIZA, PARRY, A.L.I.C.E., Cleverbot) are all below rank 20. Also "baseline approaches … out-perform" (K-means, LEM, DPEMM items), "backoff strategies" ("Pass-through: …") |
| **Exact word match missed by dense:** the evidence contains the question's key word, yet ranks low | 2 | "What is the baseline?": the evidence says "as baseline models" and is below rank 20; "What baselines are used for comparison?": "Our baseline is TransE" at rank 15 |
| **Answer word only in the section title** | 2 | "what surface-level features": evidence under "Features." at rank 7; "What baseline is used?": evidence under "Baseline Approaches" at ranks 8, 9 and below 20 |
| **A caption outranks the text evidence** | 2 | "What metrics are used for evaluation?": top 1 is "Table 4: Evaluation in supervised UD parsing. The scores are LAS."; the text evidence is at ranks 6 and 10 |
| **Needs inference or comparing numbers** | 2 | "How much training data from the non-English language…" (gold: none, a pretrained model is used) at rank 7; "which non-English language had the worst results" at rank 9 |
| Other (background or partial-answer paragraphs) | 4 | "What is the size of the Chinese data?": a second dataset paragraph not in the top 20 |

Patterns overlap (one question can show two). Some gold choices are
debatable: for "What is the baseline?" (1912.04961) the top-1 paragraph
describes two naive baselines, which a reader could accept.

### Conclusion and next step

The dense baseline is trusted: it reproduces run 001 exactly, and its
noise is about one question in 120. Next, as planned (Task 4): BM25 and
hybrid retrieval. The "exact word" misses are the direct target; the
section-title misses are the target of the later section-context
experiment. The "list items" pattern is the strongest argument seen so far
for some form of section-aware expansion (D10), but it is out of scope for
this iteration and is recorded here for later. Captions outranking text is
worth watching in the BM25 and hybrid results.

---

## 002b: Dense baseline with cached question embeddings (new reference)

- **Date:** 2026-10-31
- **Commit:** `bfbbd54` (D14: `DenseScorer` caches question embeddings)
- **Config, data, index:** as 002 (`config/composed-dense.json`,
  `data/eval-120`, `.index-full`).
- **Command:** the 002 evaluation command, run twice, writing
  `experiments/runs/002b-dense-cached.json`.
- **Results:** [experiments/runs/002b-dense-cached.json](experiments/runs/002b-dense-cached.json)

| Metric | 002 (uncached) | 002b (cached) |
| --- | ---: | ---: |
| recall@5 | 0.59 | 0.59 |
| recall@10 | 0.79 | 0.79 |
| recall@20 | 0.91 | 0.91 |

**Checks:**
- The two 002b runs produced byte-identical result files: reruns are now
  exactly repeatable.
- Prefix consistency on real data is 120 of 120 (002: 119 of 120).
- The top 5 differs from 002 on 0 of 100 answerable questions, and
  matches run 001 on 20 of 20 shipped questions. The question vectors
  cached by the first run happened to give the same rankings as 002; with
  another sample, up to about one question could have differed.

**Conclusion:** 002b replaces 002 as the dense reference. BM25 and hybrid
(003) are compared against 002b, and hybrid reuses the same cached
question vectors, so any difference comes from the change in scoring, not
from embedding noise. The failure reading in 002 still applies.

---

## 003: BM25 and hybrid (BM25 + dense) against the dense reference

- **Date:** 2026-10-31
- **Commit:** `a7cfe7b` (BM25 scorer; RRF gives no credit to a score of 0)
- **Configs:** `config/composed-bm25.json` (BM25 alone, k1 1.5, b 0.75) and
  `config/composed-hybrid.json` (dense config plus BM25, RRF k 60; nothing
  else changed). Reference: 002b.
- **Data and index:** `data/eval-120` (100 answerable), `.index-full`.
  Indexing both configs left the folder's file list unchanged; hybrid
  reused the cached paragraph and question vectors (1.5 s, no new
  embeddings), so dense and hybrid score identical question vectors.
- **Commands:** `qasper-rag index … --config config/composed-{bm25,hybrid}.json --index-dir .index-full`, then
  `python tools/retrieval_eval.py --config config/composed-{bm25,hybrid}.json --index-dir .index-full --questions data/eval-120/dev_questions.jsonl --gold data/eval-120/dev_gold.jsonl --output experiments/runs/003-{bm25,hybrid}.json`
- **Results:** [003-bm25.json](experiments/runs/003-bm25.json), [003-hybrid.json](experiments/runs/003-hybrid.json)

### Metrics

| Config | recall@5 | recall@10 | recall@20 | recall@5 = 0 | Missed at 6–10 / 11–20 / below 20 |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense (002b) | 0.59 | 0.79 | 0.91 | 34 | 25 / 17 / 26 |
| BM25 | 0.54 | 0.70 | 0.86 | 41 | 22 / 20 / 33 |
| **Hybrid** | **0.65** | 0.79 | 0.92 | 28 | 17 / 22 / 23 |

| Per question at recall@5 | Better | Worse | Same |
| --- | ---: | ---: | ---: |
| BM25 vs dense | 13 | 21 | 66 |
| Hybrid vs dense | 15 | 9 | 76 |
| Hybrid vs BM25 | 17 | 3 | 80 |

- Hybrid is never worse than both single scorers (0 questions).
- Top-k prefix consistency on real data: 120 of 120 for both configs.
- **Strength of the evidence:** hybrid's +0.06 recall@5 over dense comes
  from 15 wins against 9 losses. An exact two-sided sign test on those 24
  gives p = 0.31, so on 100 questions the gain is suggestive, not
  conclusive. Recall@10 and @20 barely move: hybrid mostly reorders the
  top of the ranking.

### Failures read

**Wins (15):** they include the "exact word" misses from 002, e.g. "What
baselines are used for comparison?" ("Our baseline is TransE"), "what
surface-level features are used?", "What metrics are used for
evaluation?", and two questions dense ranked at 7 and 9 ("How much
training data from the non-English language…", "which non-English
language had the worst results").

**Losses (all 9 read).** In each, dense had the evidence in its top 5
(ranks 1–5) and hybrid pushed it to 6–13, or out of the top 20. The cause
is the same in most of them: BM25's first choice is a paragraph that
shares only common words with the question, and RRF counts a weak rank-1
match exactly like a strong one.

| Question | Words BM25's top paragraph matched |
| --- | --- |
| did they collect their own data? | data, their, they (an Acknowledgments paragraph) |
| What is the architecture of the model? | is, model, of, the, what |
| what dataset is used? | is, what |
| What is the best performing model? | best, is, model (a table caption) |
| Does the experiments focus on a specific domain? | a, domain, focus, on, the |

In 5 of 9 losses BM25's top paragraph matched at most one content word.
The tokenizer keeps function words ("the", "is", "they"), and short
yes/no or generic questions consist mostly of them. One loss ("What are
the baselines?") is the list-items pattern from 002: four sibling
paragraphs under "Evaluation", one baseline each.

The tokenizer's plural rule also clips non-plurals ("does" → "doe",
"this" → "thi", "focus" → "focu"). It is applied the same way to
questions and paragraphs, so matching still works; it is cosmetic.

### Conclusion and next step

Hybrid is the best configuration measured so far (recall@5 0.65 vs 0.59
for dense) and is never worse than both single scorers, but the
advantage rests on 24 questions that differ and is not statistically
firm. The losses point to one specific, cheap fix: removing stopwords from
BM25 queries, so weak matches on function words stop earning top ranks.
It is not part of this iteration; it is the first candidate for the next
one, measured the same way. Section context (Task 5) is still the
planned controlled experiment for the section-title misses.

---

## 004: Section-title context (controlled experiment, D9)

- **Date:** 2026-10-31
- **Commit:** `4bd4068` (`SectionPrefixed` view and three section configs)
- **Change tested:** the indexed text of each paragraph becomes
  "section path + newline + paragraph" (e.g. "Baseline Approaches\nWe
  compare our approach to two other…"). Each section config differs from
  its raw-text config only in `view` (checked with `git diff --no-index`).
- **Data and index:** `data/eval-120` (100 answerable), `.index-full`. The
  section text was embedded once (15,757 paragraphs, 4 min, about $0.02);
  hybrid-section reused those vectors. The folder still holds only
  `paragraphs.jsonl` and `embedding-cache.sqlite`.
- **Command:** `python tools/retrieval_eval.py --config config/composed-{bm25,dense,hybrid}-section.json --index-dir .index-full --questions data/eval-120/dev_questions.jsonl --gold data/eval-120/dev_gold.jsonl --output experiments/runs/004-{…}-section.json`
- **Results:** [004-bm25-section.json](experiments/runs/004-bm25-section.json), [004-dense-section.json](experiments/runs/004-dense-section.json), [004-hybrid-section.json](experiments/runs/004-hybrid-section.json)

### Metrics (raw text → section view)

| Scorer | recall@5 | recall@10 | recall@20 | Better / worse at recall@5 | Sign test p |
| --- | --- | --- | --- | --- | ---: |
| BM25 | 0.54 → 0.54 | 0.70 → 0.71 | 0.86 → 0.89 | 3 / 3 | 1.00 |
| Dense | 0.59 → 0.60 | 0.79 → 0.78 | 0.91 → 0.92 | 9 / 8 | 1.00 |
| Hybrid | 0.65 → 0.66 | 0.79 → 0.80 | 0.92 → 0.94 | 7 / 6 | 1.00 |

### Failures read (all 17 losses)

**Wins** include the cases the view targeted: "What baseline is used?"
(evidence under "Baseline Approaches") improves with every scorer, and
"what surface-level features are used?" and "How do the backoff
strategies work?" improve with dense.

**Losses** follow two patterns:
- **A matching title lifts the whole section.** Every paragraph in a
  section gets the same prefix, so when the title matches the question,
  all its paragraphs rise together and push the evidence elsewhere down.
  "How do they match annotators to instances?": the evidence is under
  "Routing To Experts or Crowd" and drops to rank 9–11, below paragraphs
  under "Involving Expert Annotators".
- **Evidence in generically named sections.** In 7 losses the evidence
  sits under "Introduction" or a long paper-specific title that shares
  nothing with the question, while other paragraphs gain from titles such
  as "Abstract", "Dataset description" or "Results and Discussion".
  "How many tweets are in the dataset?": the evidence (Introduction) drops
  to rank 6 behind the Abstract.

### Conclusion

No net effect: each scorer gains and loses about the same number of
questions, and recall@5 changes by at most +0.01, within what one
question moves. The view does what it was designed to do for its target
cases, but the same mechanism costs other questions. **D9: not adopted.**
`SectionPrefixed` stays available for later combinations. Recall@20 rises
slightly with the view (hybrid 0.92 → 0.94), which only matters if a
reranker is ever tested (D7).

**Best configuration so far: raw-text hybrid** (003; recall@5 0.65,
@10 0.79, @20 0.92).

---

## 005: Chosen retriever end to end (hybrid, shipped 20 questions)

- **Date:** 2026-10-31
- **Commit:** `0f97f56` (no code changes since 003)
- **Config:** `config/composed-hybrid.json` with the starter's generator
  (unchanged prompt, `openai/gpt-6-luna`, temperature 0), index
  `.index-full`. Compared with 001 (starter baseline, same 20 questions).
- **Commands:** `qasper-rag predict --index-dir .index-full --questions results/hybrid-20/questions.jsonl --config config/composed-hybrid.json --output results/hybrid-20/predictions.jsonl`
  (run in two batches of 10 with a 65 s pause, see below), then
  `python tools/score.py` and
  `python tools/error_analysis.py --run-dir results/hybrid-20 --output experiments/runs/005-hybrid-20.md`.
- **Results:** [005-hybrid-20.md](experiments/runs/005-hybrid-20.md),
  [005-hybrid-20.metrics.json](experiments/runs/005-hybrid-20.metrics.json)
- **Rate limit:** the first attempt in one batch hit OpenRouter's limit for
  new accounts (20 requests per minute for this model): one question
  failed with HTTP 429 after retries and was scored as missing. That run
  was discarded and the 20 questions were rerun in two batches of 10, with
  a 65 s pause before each, and the predictions merged. Latency per
  question is unaffected (pauses happen between `predict` runs).

### Metrics

| Metric | 001 (starter) | 005 (hybrid) |
| --- | ---: | ---: |
| paragraph_evidence_recall_at_5 | 0.33 | **0.63** |
| citation_evidence_f1 | 0.27 | **0.43** |
| answerable_question_coverage | 0.73 | 0.87 |
| answer_token_f1 | 0.22 | 0.24 |
| abstention_precision / recall | 0.20 / 0.20 | 0.33 / 0.20 |
| mean_query_latency_ms | 2,031 | 1,792 |

| Outcome (error report) | 001 | 005 |
| --- | ---: | ---: |
| retrieval_miss | 9 | 4 |
| retrieval_partial | 2 | 3 |
| retrieved_answered | 3 | 6 |
| over_abstain | 1 | 2 |
| missed_abstain | 4 | 4 |
| correct_abstain | 1 | 1 |

**Consistency check:** the top 5 that `predict` used equals the top 5 in
the retrieval evaluation (003-hybrid) on 15 of 15 answerable questions,
and recall@5 is 0.63 in both, so the retrieval-only numbers reflect what
the application does.

### Failures read

- **Retrieval gains carry through:** "What baselines are used for
  comparison?" is now answered "TransE" with the right citation, and
  "what surface-level features are used?" gives the gold features.
- **The generator is now the bottleneck.** Answer F1 moved only 0.22 →
  0.24 although evidence recall almost doubled:
  - **Hedged answers instead of abstaining:** the same 4 unanswerable
    questions as in 001 get answers like "…the passages do not say that
    the reward is made available at each step" (D12, not addressed yet).
  - **Wordy answers:** "Paper 1611.08661 uses TransE as the baseline."
    where the gold is "TransE"; "Yes. Paper 1705.01265 reports…" where it
    is "Yes" (D12).
  - **Abstains with the evidence retrieved (2):** "Is the lexicon the same
    for all languages?" (gold Yes; also in 001), and "How much training
    data from the non-English language…", whose gold "No data. Pretrained
    model is used." needs an inference the model declines to make.
  - **Wrong reasoning over numbers (1):** "which non-English language had
    the worst results?" now retrieves the right paragraph but answers
    "Russian" (gold "Turkish").
- **Caveat:** one end-to-end run on 20 questions. In 000 vs 001 the same
  config gave different answer text on 6 of 10 questions, so answer F1
  and abstention numbers here are noisy; the retrieval and citation
  numbers are not (retrieval is repeatable, D14).

### Retrieval conclusion for this iteration

- **Chosen:** raw-text hybrid BM25 + dense (D15). Recall@5 0.65 / @10 0.79
  / @20 0.92 on 100 dev questions.
- **Not adopted:** section-title context (D9, 004).
- **Remaining retrieval failures:** BM25's weak function-word matches
  (most hybrid losses, 003); list items under one section (D10 notes);
  captions outranking text; questions needing inference.
- **Next, by expected impact:** the generator fixes in D12 (hedging,
  wordiness), since they now limit the end-to-end score; then stopword
  removal for BM25; then, if justified, a reranker test (D7 headroom) or
  section-aware expansion for lists (D10).

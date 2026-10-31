# Evaluating Your Agent (Optional)

This page is optional. You can evaluate your agent with the questions, gold answers and tools described here, or with your own scripts, metrics and test questions.

In many projects there is almost no labeled data, because labels are expensive to produce. This dataset has human labels, so you can measure your agent instead of judging it by reading a few answers. For each question, the gold file gives one or more reference answers and the paragraph IDs that support each answer. You can ignore this data, but you need some way to show whether a change to your agent helped.

## How the Evaluation Is Set Up

Four small tools in `tools/` work together. They only read JSONL files, so they work with any agent that writes predictions in the [output format](baseline.md#interfaces-and-output-contract).

| Tool | What it does |
| --- | --- |
| `tools/score.py` | The main scorer. It checks every prediction against the output rules, then compares it with the gold labels. It needs no API key. See the metric table below. |
| `tools/ragas_eval.py` | Scores answerable questions with [RAGAS](https://docs.ragas.io). A language model judges whether each answer is faithful to the retrieved paragraphs, whether retrieval found the right paragraphs, and whether the answer matches the reference. It calls the model API, so it costs a little. |
| `tools/subset.py` | Copies the first N questions, their gold labels, and only their papers' paragraphs, so a test run is cheap. |
| `tools/run_eval.py` | Runs subset, index, predict, score and optionally RAGAS in one command. It is built for the baseline, so for your own agent use the other tools directly. |

`score.py` counts matching words and paragraph IDs, so it is fast and the same every time, but it misses answers that are right in different words. RAGAS compares meaning, but it uses a model, so its scores vary a little from run to run. The two are meant to be read together.

## How to Use the Data to Evaluate Your RAG Agent

The corpus is what your agent searches, the questions are its input, and the gold file is only used afterward to score the output. Never load the gold file in your agent, in a prompt, or in the index.

Each QASPER question is about one paper, named by its `paper_id`, and its gold evidence comes from that paper only. Your agent should answer across all papers, but when you evaluate with this data, search only the question's paper. Otherwise the agent may cite a correct paragraph from another paper that the gold does not list, and the scorer rejects retrieval outside the question's paper.

1. Build the index from `data/corpus.jsonl`.
2. Run your agent on `data/dev_questions.jsonl` and write one prediction per question to a JSONL file. Each prediction needs the fields in the [output contract](baseline.md#interfaces-and-output-contract).
3. Score the predictions against the gold file with `tools/score.py`. It prints the metrics and writes them to the file you give with `--output`.
4. Change one thing in your agent, run it on the same questions again, and score it again.
5. Compare the two metric files, and then read individual questions to see why the numbers moved.

```bash
python tools/score.py --corpus data/corpus.jsonl \
  --questions data/dev_questions.jsonl --gold data/dev_gold.jsonl \
  --predictions results/my_run.jsonl --output results/my_run_metrics.json
```

The scorer rejects a prediction file that breaks the contract, e.g., a citation quote that is not an exact piece of the cited paragraph. A question with no prediction counts as a zero, so a crash cannot make a run look better.

## What Each Metric Tells You

| Metric | What it measures |
| --- | --- |
| `answer_token_f1` | How many words your answer shares with the best matching reference answer, after lowercasing and removing punctuation and the words a, an and the. 1.0 is an exact match. |
| `paragraph_evidence_recall_at_5` | For answerable questions, the share of the gold evidence paragraphs that are in your retrieved list of up to five paragraphs. This tells you whether retrieval found the right text. |
| `citation_evidence_f1` | For answerable questions, how closely the paragraphs you cited match the gold evidence paragraphs. It is lower if you cite too many or too few. |
| `abstention_precision` | Of the questions where your agent abstained, the share that really have no answer in the paper. |
| `abstention_recall` | Of the questions that really have no answer, the share where your agent abstained. |
| `abstention_f1` | One number that combines the two abstention scores above. |
| `answerable_question_coverage` | Of the questions that have an answer, the share where your agent gave an answer instead of abstaining. |
| `missing_outputs` | How many questions had no prediction. |
| `mean_query_latency_ms` | Average time to answer one question, taken from the `latency_ms` field of each prediction. |

When a question has several reference answers, the scorer uses the best match for each metric separately.

## Worked Example with 10 Questions

This example uses the first 10 development questions, which keeps the cost small. `tools/subset.py` copies those questions, their gold labels, and only the paragraphs of their 10 papers into a folder, so indexing is a small fraction of the full build. Retrieval is limited to one paper per question, so the answers match what the full corpus would give.

```bash
python tools/subset.py --count 10 --output-dir results/ragas-10
qasper-rag index --corpus results/ragas-10/corpus.jsonl \
  --config config/baseline.json --index-dir .index-ragas10
qasper-rag predict --index-dir .index-ragas10 \
  --questions results/ragas-10/questions.jsonl \
  --config config/baseline.json --output results/ragas-10/baseline.jsonl
python tools/score.py --corpus results/ragas-10/corpus.jsonl \
  --questions results/ragas-10/questions.jsonl --gold results/ragas-10/gold.jsonl \
  --predictions results/ragas-10/baseline.jsonl \
  --output results/ragas-10/baseline_metrics.json
```

You can also score with [RAGAS](https://docs.ragas.io), which uses a language model as the judge and compares meaning instead of counting shared words. RAGAS is an optional extra, so install it into the same environment first with `python -m pip install --editable ".[ragas]"`. `tools/ragas_eval.py` only reads the JSONL files. It scores answerable questions only, because an unanswerable question has no reference answer. Use the abstention metrics from `tools/score.py` for those.

```bash
python tools/ragas_eval.py \
  --corpus results/ragas-10/corpus.jsonl --questions results/ragas-10/questions.jsonl \
  --gold results/ragas-10/gold.jsonl --predictions results/ragas-10/baseline.jsonl \
  --output results/ragas-10/baseline_ragas.json
```

To run all of these steps with one command, use `tools/run_eval.py`. It writes its results to `results/<config name>-<count>` and accepts `--ragas` to also run RAGAS.

```bash
python tools/run_eval.py --count 10 --config config/baseline.json
```

Treat any 10 question result as a smoke test. Model output changes from run to run even at temperature 0, so your numbers will differ.

One run of the baseline config on these 10 questions gave the results below. They show what to expect from the tools and are not a target to beat.

| Metric | Result |
| --- | --- |
| `answer_token_f1` | 0.13 |
| `paragraph_evidence_recall_at_5` | 0.28 |
| `citation_evidence_f1` | 0.28 |
| `answerable_question_coverage` | 0.67 |
| `abstention_recall` | 0.00 |
| `mean_query_latency_ms` | about 3,100 |
| RAGAS `faithfulness` | 0.56 |
| RAGAS `context_recall` | 0.50 |
| RAGAS `answer_correctness` | 0.22 |

The scores are low because the baseline is simple. It embeds the question once, takes the five closest paragraphs, and has no reranking or evidence check. Evidence recall of 0.28 means retrieval often misses the paragraphs the annotators chose, and a coverage of 0.67 means the baseline declines to answer one question in three that have an answer. The abstention recall of 0.00 comes from the one unanswerable question in this slice, which the baseline answered instead of declining. RAGAS `answer_relevancy` is not listed. It asks the judge model for three generations per question and the provider returned one, so that score is not reliable here.

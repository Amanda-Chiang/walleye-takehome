# The Papers and the Data

## What is QASPER?

[QASPER](https://aclanthology.org/2021.naacl-main.365/) is a research-paper
question-answering dataset with **5,049 questions about 1,585 NLP papers**.
Question writers saw a paper's title and abstract and asked questions whose
answers might be in the full text. Other annotators supplied answers and
supporting evidence. Questions include extractive, free-form, yes/no, and
unanswerable cases.

The original dataset has three splits:

| Original split | Papers | Questions | Usual purpose |
| --- | ---: | ---: | --- |
| Train | 888 | 2,593 | Development and tuning |
| Validation (`dev`) | 281 | 1,005 | Final comparison on a fixed sample |
| Test | 416 | 1,451 | Final, held-out evaluation |

The original QASPER records put the paper text, the questions, the answers and
the evidence together. The [official dataset card](https://huggingface.co/datasets/allenai/qasper)
describes them. A paper has an ID, a title, an abstract, and full text in
sections and paragraphs. It may also have figures and tables. Each question has
reference answers and evidence, and sometimes several annotators gave different
ones.

**In this exercise the paper text, the questions and the labels have separate
roles.** Do not use the test split. Do not put answers or evidence labels into
the searchable corpus or into prompts, and do not hard-code answers.

## What Data Do You Receive?

The papers are the main part of this exercise. Your agent reads the papers, finds evidence in them, and answers questions about them. The questions and gold answers are only a way to measure how well your agent does that. They are not part of the agent, and your agent must work on other papers and other questions without code changes.

| File | What it contains | How to use it |
| --- | --- | --- |
| `data/corpus.jsonl` | 15,757 paragraphs from all 281 QASPER `dev`/validation papers, with stable paragraph IDs and paper IDs | The knowledge base to index and search; contains no QA labels |
| `data/dev_questions.jsonl` | A small sample of 20 validation questions, each with `question_id`, `paper_id`, and `question` | Inputs to the RAG agent; contains no answers |
| `data/dev_gold.jsonl` | Human reference answers and evidence paragraph IDs for those 20 questions | Scoring and error analysis only; never load it during inference |
| `data/manifest.json` | Source checksum, counts, and subset selection rules | Provenance and reproducibility |
| `tests/fixtures/` | A tiny synthetic corpus, two questions, gold, and valid predictions | Learn the schemas and verify the setup cheaply |

The corpus is what your agent searches. The question file is the list of
questions it answers. The gold file is the answer key, and only the scorer uses
it. Your agent writes a file of predictions, and the scorer compares that file
with the gold. Some questions have more than one acceptable answer, and the
scorer keeps all of them.

A corpus row looks like this synthetic example:

```json
{"paragraph_id":"demo:p00000","paper_id":"demo","section":"Results","text":"The model achieved 82% accuracy on the evaluation set."}
```

Its question might be:

```json
{"question_id":"demo-answer","paper_id":"demo","question":"What accuracy did the model achieve?"}
```

A supported answer is `82%`, citing `demo:p00000` with the quote `82% accuracy`.
If someone asks how many GPUs were used and the retrieved paragraphs do not say,
the agent should abstain, even if it could guess a likely number. When the agent
abstains, it should still list the paragraphs it searched, so a reviewer can
see what it looked at.

## What Kinds of Questions Should the Agent Answer?

Questions are about a paper's methods, datasets, experiments, comparisons,
results and limitations. They can be short and leave out details, because the
paper gives the context. Here are examples from the development questions:

- “What is the baseline?”
- “How much training data from the non-English language is used by the system?”
- “Do they report results only on English datasets?”

Expect four kinds of questions: factual answers copied from the text, short
explanations that combine several paragraphs, yes/no questions, and questions
the paper does not answer. A relevant paragraph may use different words from the
question. Answer yes/no questions with `Yes` or `No`, and keep other answers
short. A paragraph that only mentions the topic is not enough. It must support
the answer.

## Scope and Limitations of This Subset

The corpus has titles, abstracts, section paragraphs and raw figure and table
captions. It is parsed text, so it may contain citation markers such as
`BIBREF16` and imperfect math. This is a **text-only** task. You do not need to
read PDFs or rebuild tables and figures. Keep the paragraph IDs and the text
unchanged when you split or embed the papers, and map your evidence back to
them before you return it. The corpus has captions but not the figures
themselves, so explain in your writeup how you handle evidence from figures and
tables.

The sample has 20 questions: 15 with an answer and five without. The manifest
records the source file checksum and how the questions were chosen. They were
picked by a fixed hash of each question ID, with fixed counts of answerable and
unanswerable questions, and not by how well any system did on them. We left out
questions where the annotators disagreed on whether there is an answer, where
the evidence is a figure or table or cannot be matched to a paragraph, or where
an annotation needs more than five evidence paragraphs. For the questions we
kept, we kept every annotation. If the same text appears in two paragraphs, the
evidence points to the first one. This small sample is a quick check. For a
stronger comparison, choose more questions with `tools/prepare_data.py --count`,
or write your own.

You can use the QASPER training papers while you build your agent. Compare
versions of your agent on the same questions. Use the gold only for scoring and
for studying failures. Your agent must work with a different question file
without code changes. A reviewer may give it other papers and keep the gold
answers private. Tell us if you use a model that was trained on QASPER or on
this benchmark. The data is shared under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Getting the Data from the Source

You do not need to download anything to start, because the files in `data/` are already prepared. Read this section if you want to check where they came from, rebuild them, or use other QASPER papers.

The source is QASPER version 0.3 from the Allen Institute for AI. Two scripts turn it into the files above (`just data` runs both). The first one needs an internet connection.

1. `tools/download_data.py` downloads the QASPER archive from the dataset's public bucket and copies the dev JSON file into the folder you choose. The archive is about 11 MB, and the dev file contains the 281 validation papers.

   ```bash
   python tools/download_data.py --data-dir data --splits dev
   ```

2. `tools/prepare_data.py` reads the dev JSON and writes `corpus.jsonl`, `dev_questions.jsonl`, `dev_gold.jsonl`, and `manifest.json`. The choice of questions is fixed by a hash of each question ID, so the output is the same every time.

   ```bash
   python tools/prepare_data.py --source data/qasper-dev-v0.3.json --output-dir data
   ```

After step 2, the `source_sha256` value in `data/manifest.json` should match the one in this repository. We checked that both steps rebuild the shipped files exactly.

Add `--splits train` to also download the training papers. Do not download or use the test split, even though `--splits test` is possible.

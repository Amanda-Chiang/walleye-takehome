# Special Projects Take-Home Assignment: Build a RAG Agent for QASPER

## Task Details

Build a RAG agent that answers questions about a set of scientific papers.
Given a question, the agent should find evidence in the papers and return an
answer with citations that a reviewer can check. When the papers do not
contain enough evidence, it should say so instead of guessing.

The agent should answer across all the papers, not just one. A question may
need evidence from several papers, or the user may not know which paper has
the answer. The QASPER questions and gold answers are different: each one is
about a single paper, and its gold evidence comes from that paper only. When
you evaluate with them, limit the search to the question's `paper_id`, so the
results compare fairly with the gold. The baseline does both: the chat page
and `qasper-rag answer` search every paper, and evaluation runs search only
the question's paper.

How you build it is up to you: how you parse and chunk the papers, how you
retrieve evidence, how you write and check answers, and how you evaluate and
improve the agent. Explain the choices you make and their tradeoffs. Give a
reviewer some way to ask questions and see the answers and citations, e.g. a
command-line tool, a web page or an MCP server.

You do not need to train a model, use a GPU or reach a particular score.

## Expected Agent Behavior

A question goes in, and an answer with evidence comes out. One way to split the
work is:

1. **Prepare the papers once.** Parse them, split them into chunks, and build
   whatever index your retrieval needs. Keep each chunk's paragraph ID.
2. **Take a question.** It may name one paper, several papers, or none, in
   which case the agent searches every paper.
3. **Find evidence.** Retrieve the paragraphs most likely to answer it.
4. **Decide.** If the paragraphs support an answer, write it and quote the
   evidence. If they do not, abstain.
5. **Return the result:** the answer or an abstention, the quotes with their
   paragraph IDs, and the paragraphs you searched.

You choose the design. This is only a way to talk about the parts.

## Examples

**One paper.** The question comes with a paper ID.

> **Question** (paper `1705.01265`): Do they report results only on English datasets?
>
> **Answer:** Yes
>
> **Citation:** paragraph `1705.01265:p00008`, quote "For all the tasks in our experimental study, we use 36 millions English tweets"

**Across papers.** The question has no paper ID, so the agent searches every paper and says which paper each claim comes from. This example is illustrative. We wrote the question, and the quotes are real text from the corpus.

> **Question:** Which papers use pre-trained GloVe embeddings, and for what?
>
> **Answer:** Paper `1707.06806` uses GloVe vectors to initialize the embedding layer. Paper `1808.10245` uses GloVe for word-level features. Paper `1901.01010` uses 50-dimensional GloVe embeddings to represent words.
>
> **Citations:**
> - `1707.06806:p00012`, "pre-trained GloVe word vectors BIBREF16 to initialize the embedding layer"
> - `1808.10245:p00017`, "A pre-trained GloVe BIBREF17 representation is used for word-level features."
> - `1901.01010:p00022`, "We use the pre-trained GloVe BIBREF39 50-dimensional word embeddings to represent words."

**Abstaining.** For "Is the discriminator's reward made available at each step to the generator?" (paper `1908.07195`), the annotators found no answer in the paper. A good agent answers `Unanswerable` and cites nothing, even if it could guess.

**A bad answer.** An answer that sounds right but has no matching quote is a failure, even when it happens to be correct. For example, "Yes, the paper uses only English data" with a made-up quote like "all experiments are in English" fails, because that sentence is not in the paragraph. Every quote must appear exactly in the cited paragraph.

## Expected Output

Each answer is a record like this. The full rules are in
[docs/baseline.md](docs/baseline.md#interfaces-and-output-contract), and
`tools/score.py` reads this format.

```json
{
  "question_id": "example-1",
  "answer": "Yes",
  "abstain": false,
  "citations": [{"paragraph_id": "1705.01265:p00008", "quote": "we use 36 millions English tweets"}],
  "retrieved_paragraph_ids": ["1705.01265:p00008", "1705.01265:p00007"],
  "latency_ms": 1200.0
}
```

An abstention has `"answer": "Unanswerable"`, `"abstain": true` and no
citations. You do not have to use this format. If you choose your own, add a
short converter to this format, or write your own scoring.

## Getting Started

1. **Get the data.** The papers are the QASPER v0.3 validation papers: 15,757
   paragraphs from 281 NLP papers. They come prepared in `data/corpus.jsonl`.
   To download the original QASPER files instead, or to get the training
   papers too, run `python tools/download_data.py --splits dev train`. See
   [docs/data.md](docs/data.md) for the file formats and how the data was
   prepared.
2. **Choose your models.** To run the baseline, get an
   [OpenRouter](https://openrouter.ai) API key; the baseline is set up for
   OpenRouter. For your own implementation, use any provider, free models or
   local models you like.
3. **Decide how much of this repository to use.** Everything except the
   papers is optional:
   - **The baseline** in `src/qasper_rag/` is a working reference. You can
     build on it, change it, or ignore it and start from scratch with your own
     design, language and libraries. See [docs/baseline.md](docs/baseline.md).
   - **The questions and gold answers** in `data/` are there if you want them
     for evaluation. You can use them with `tools/score.py` and the optional
     RAGAS scoring, or write your own evaluation scripts, metrics and test
     questions. See [docs/evaluation.md](docs/evaluation.md).

## Rules

- Any AI coding assistant, model provider and framework is fine.
- You need to build the RAG system yourself. You are free to use frameworks and libraries or a vector database, and to host parts of it in the cloud. Do not use a managed service that does the whole pipeline for you, i.e. one that takes your documents and returns cited answers, such as OpenAI file search, Amazon Bedrock Knowledge Bases or Vertex AI Search. If you do use one, say so in your writeup. We will then weigh your evaluation much more heavily, because the evaluation is the main thing we can judge.

## Evaluation Criteria

We look at how you think and how you work. We look at these things:

- **Design.** Your architecture, how you split the problem into parts, and
  whether you can explain the tradeoffs of your choices, e.g. chunking,
  retrieval and how you write answers.
- **Evaluation.** How you measure your agent and how you use the results to
  improve it. We value looking at real failures before choosing metrics: read
  your agent's wrong answers, group them by cause, and fix the most common
  cause first. We value evaluating retrieval and answers separately, so you
  can tell which part of the system caused a bad answer. If you use a model to
  judge answers, check its judgments against your own reading first. Be
  careful with a small test set, where one question can move the score, and be
  honest about what the numbers can and cannot show.
- **Engineering.** Code that someone else can run and read, handling of
  failures, and tests where they help.
- **Communication.** A clear writeup, and your honesty about limits and about
  what you left out.

**We do not judge your choice of models or the look of the interface, and we do
not expect you to pay for models to complete this case study.** The interface
only needs to be good enough for testing. We do not score the final numbers
against a target, so a clear result that is worse than you hoped is fine if
you explain why.

Be ready to explain every part of your submission, including code an AI tool
wrote, and to change it in the follow-up conversation.

## Submission Details

Upload a zip containing -

- your source code, with setup instructions;
- a **WRITEUP.md** report, described below;
- any other supporting material you think is important, e.g. evaluation
  results, experiments or ablation studies.

The WRITEUP.md should be 1 to 2 pages and cover:

- your architecture and design choices, i.e. chunking, retrieval, citations,
  abstention, and how you handle evidence from figures and tables;
- how you evaluated the agent, where it works well, and where it fails;
- the models and providers you used, with their versions and the approximate
  cost;
- the time you spent, what you left out, and which AI tools you used, what you
  gave them to do, and how you checked their work.

A follow-up conversation will discuss your implementation.

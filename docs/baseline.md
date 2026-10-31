# The Baseline (Optional Reference)

This page covers the reference implementation in `src/qasper_rag/`. You do not have to use it. If you build your own agent, the [output contract](#interfaces-and-output-contract) is still useful, because `tools/score.py` reads that format.

## Set Up and Run the Starter

### 1. Create a Python Environment

Open a terminal in the extracted starter directory or repository root. Use
**Python 3.10 or later** (3.13 is recommended). Runtime code uses the standard library; install the project in
editable mode to install `qasper_rag` and register its `qasper-rag` command.
On macOS/Linux:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --editable .
python -m unittest discover -s tests -v
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv` and
activate it with `.venv\Scripts\Activate.ps1`. The remaining commands use `python`;
then run `python -m pip install --editable .`. Put multiline shell examples on one
line in PowerShell. See the
[Python venv documentation](https://docs.python.org/3/library/venv.html).
If you prefer uv, follow its [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
and use `uv venv --python 3.13` to create the environment, activate it, and run
`uv pip install --editable .`.

Tests use a temporary localhost fake provider and never contact a hosted model.
They check retrieval scope, embedding caching, output validation, custom classes,
missing-output penalties, and the command-line pipeline.

### 2. Configure Model Access

The starter defaults to OpenRouter's [authenticated API](https://openrouter.ai/docs/quickstart)
for embeddings and answers. Use your own API key. Local models and other hosted
providers are fine if you can explain the choice.

```bash
export OPENROUTER_API_KEY='<your key>'
```

PowerShell equivalent: `$env:OPENROUTER_API_KEY = '<your key>'`.
Never commit or submit the key. `.env.example` lists variable names only;
`.env` files are not automatically loaded.

| Stage | Default | What it does |
| --- | --- | --- |
| Embeddings | `openai/text-embedding-3-small` | Embeds corpus paragraphs and each question into 256-dimensional vectors |
| Retrieval | Cosine similarity within the chosen paper, or across all papers | Selects the top five paragraphs |
| Answer generation | `openai/gpt-6-luna` | One call generates a grounded answer or an abstention with citations |

The baseline has no reranking, keyword search, query rewriting, or evidence check.
This is on purpose, so there is room to improve it.

Settings live in `config/baseline.json`. Optional environment overrides:

| Variable | Overrides |
| --- | --- |
| `RAG_MODEL` | Answer model |
| `RAG_EMBEDDING_MODEL` | Embedding model; rebuild the index after changing it |
| `RAG_BASE_URL` | API base URL for all stages; defaults to `https://openrouter.ai/api/v1` |
| `RAG_API_KEY` | Credential; takes precedence over `OPENROUTER_API_KEY` |

Clear old overrides when you want to use the model IDs in the config file. The
baseline never switches to another model or provider by itself.

### 3. Run the Tiny Example First

This embeds three synthetic paragraphs and answers two questions through the
full pipeline. It incurs small API usage and checks access before a larger index.

```bash
qasper-rag index --corpus tests/fixtures/corpus.jsonl \
  --config config/baseline.json --index-dir .index-smoke
qasper-rag predict --index-dir .index-smoke \
  --questions tests/fixtures/questions.jsonl --config config/baseline.json \
  --output results/smoke.jsonl
python tools/score.py --corpus tests/fixtures/corpus.jsonl \
  --questions tests/fixtures/questions.jsonl --gold tests/fixtures/gold.jsonl \
  --predictions results/smoke.jsonl --output results/smoke_metrics.json
```

To check the file contract and scorer without any API access, score the supplied
`tests/fixtures/predictions.jsonl` instead. Its perfect metrics and zero latency are
illustrative; they are not measured model performance.

### 4. Build the Development Index Once

```bash
qasper-rag index --corpus data/corpus.jsonl \
  --config config/baseline.json --index-dir .index
```

Indexing embeds the corpus and saves vectors, metadata, and a content-addressed
SQLite cache under `.index/`. It uses the API and may take several minutes.
Completed embedding batches survive a failed run: rerun the same command to reuse
them. Neither index construction nor index loading counts toward query latency.

Changing the corpus, embedding model, dimensions, or embedding-text truncation
requires rebuilding the index. The runner rejects incompatible artifacts. Changing
only retrieval logic, prompting, or answer generation can reuse the index.

### 5. Open the Local HTML Chat Page

After building `.index`, start the local web server:

```bash
qasper-rag serve --index-dir .index --config config/baseline.json
```

Open `http://127.0.0.1:8000` in a browser. Choose "All papers" or one of the
indexed papers, ask a question, and expand the citation and retrieval panels to
inspect the supporting quotes and retrieved paragraph IDs. With "All papers" the
answer can compare several papers, and each citation names the paper it came from.
Start a fresh chat to clear the visible transcript. Each question is answered
independently against the selected scope; the page does not pass earlier turns as
answer context. Press `Ctrl+C` in the terminal to stop the server.

The page is served locally and bound to `127.0.0.1` by default. Model calls still
go to the configured API providers, which receive the question and retrieved
paragraph text. Use public exercise data only. The UI requires an existing index;
it does not run evaluation or read development gold. Use the commands below
to compare versions with the scorer.

### 6. Compare Baseline and One Change

Copy `config/baseline.json` to a new file, e.g. `config/my-change.json`, and
change it or point it at your own class (see "Extending the Starter"). Then run
both configs on the same questions and score both.

If your change only touches retrieval logic, prompts or the generator, both
runs can use the same `.index`. If it changes the embedding settings, build a
separate index for it first:

```bash
qasper-rag index --corpus data/corpus.jsonl \
  --config config/my-change.json --index-dir .index-my-change
```

Then run and score both:

```bash
qasper-rag predict --index-dir .index \
  --questions data/dev_questions.jsonl \
  --config config/baseline.json --output results/baseline.jsonl
qasper-rag predict --index-dir .index-my-change \
  --questions data/dev_questions.jsonl \
  --config config/my-change.json --output results/my-change.jsonl

python tools/score.py --corpus data/corpus.jsonl \
  --questions data/dev_questions.jsonl --gold data/dev_gold.jsonl \
  --predictions results/baseline.jsonl --output results/baseline_metrics.json
python tools/score.py --corpus data/corpus.jsonl \
  --questions data/dev_questions.jsonl --gold data/dev_gold.jsonl \
  --predictions results/my-change.jsonl --output results/my-change_metrics.json
```

Use `--index-dir .index` for both `predict` commands if you did not build a
second index. Show when your change helps and when it adds latency or fails. A
negative result is fine if you explain it. Use the same questions for both runs
and list every difference. Script names can change, but document the commands.

Each baseline query makes one API call to embed the question and one to answer
it. Retries, and one regeneration after an invalid model output, add calls.
Query latency includes every query-time stage and retry. Before a full run,
check current model pricing and estimate the cost from actual token usage.
Report the approximate cost, the models, and the calls per question with your
metrics. The [embedding model page](https://openrouter.ai/openai/text-embedding-3-small)
lists current pricing and limits.

### Shortcuts with just

If you have [just](https://just.systems) installed, you can use these shortcuts. Activate your virtual environment first, because the recipes use the `python` and `qasper-rag` commands from it.

| Recipe | What it does |
| --- | --- |
| `just data` | Downloads the dev papers and rebuilds the data files |
| `just index build` | Builds the index of every paper |
| `just app start` | Starts the chat page |
| `just eval run 10` | Scores the first 10 dev questions |
| `just eval ragas 10` | The same, plus RAGAS scoring |
| `just test` | Runs the tests |

The app and eval recipes build the index first, using `.index`, and the first build of the full corpus can take a few minutes. To use a smaller corpus or another index folder, set `QASPER_CORPUS` or `QASPER_INDEX`, e.g. `QASPER_INDEX=.index-small just index build`. Run `just --list --list-submodules` to see every recipe. The justfile only wraps the commands on this page, so you do not need just.

### Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing credential | Export your API key in the same terminal before running |
| HTTP 401/402 | Check your key and your account credits |
| HTTP 403 | Read the provider message. Some free models only work with certain apps |
| HTTP 404 mentioning privacy, data retention or guardrails | Your account settings block that model's provider. Pick another model or change the setting |
| HTTP 429/5xx | The baseline retries a few times. If it keeps failing, wait and try again |
| TLS certificate error | Configure a trusted CA bundle through `SSL_CERT_FILE`; never disable verification |
| Missing/incompatible embedding index | Rebuild with the same corpus and embedding config used by prediction |
| Invalid JSON from the model | Look at the failed output and fix the cause. Do not turn provider errors into abstentions |
| Answer breaks the output contract | The answer is regenerated once. If the second answer also breaks the contract, the run fails with the reason. |
| Citation quote is not an exact substring | A quote that is almost exact, e.g. a dropped closing `$` in math, is automatically replaced with the matching text from the paragraph (`src/qasper_rag/validation/quotes.py`). A quote that does not closely match any text in the cited paragraph still fails the run, because the answer is not grounded. |

We ran the full baseline on the two synthetic example questions, which gave a
supported answer with a valid citation and an abstention. This checks that
the setup and the output format work. It says nothing about quality on the real
questions.

## Interfaces and Output Contract

The `qasper-rag answer` command takes a question, and optionally a paper ID,
and prints one JSON object. It holds the answer, whether the question was
answerable, the citations, the retrieved paragraph IDs, the latency and a short
trace. The chat page gets the same fields, plus `question_id`, `abstain` and
the section of each citation. Run one answer with:

```bash
qasper-rag answer --index-dir .index --config config/baseline.json \
  --paper-id <paper_id> --question "<question>"
```

Leave out `--paper-id` to search every indexed paper, or repeat it to search several.

The response looks like this:

```json
{
  "paper_id": "demo",
  "question": "What accuracy did the model achieve?",
  "answer": "82%",
  "answerable": true,
  "abstain": false,
  "citations": [
    {"paragraph_id": "demo:p00000", "quote": "82% accuracy"}
  ],
  "retrieved_paragraph_ids": ["demo:p00000", "demo:p00001"],
  "latency_ms": 123.4,
  "trace": [
    "Retrieved 2 passages from 1 paper(s).",
    "Answered with 1 source citation(s)."
  ]
}
```

The values above are only an example. The baseline measures `latency_ms` when it
runs, and the trace lists the steps it took. The `answer` command and the chat
page also add fields such as `retrieved_sources`, `paper_scope` and
`papers_searched`. The `predict` command writes only the scorer fields.

`qasper-rag predict` writes one JSON object per input question to a JSONL file.
Each object must follow the table below, and you may add extra fields. See
`tests/fixtures/predictions.jsonl` for an answer and an abstention.

| Field | Required value |
| --- | --- |
| `question_id` | The ID of the question, with no repeats and no unknown IDs |
| `answer` | A nonempty string. On an abstention it is exactly `Unanswerable` |
| `abstain` | `false` for an answer, `true` when the evidence is not enough |
| `citations` | Up to five objects with **only** `paragraph_id` and `quote`, each paragraph at most once. At least one for an answer, none for an abstention |
| `retrieved_paragraph_ids` | The paragraphs you searched, best first: up to five different IDs from the question's paper. Keep them when you abstain |
| `latency_ms` | A number of milliseconds, zero or more, for the time to answer one question, including retries. Do not count indexing or loading the index |

Each quote must be a nonempty piece of the cited paragraph, copied exactly,
with the same capital letters. Every cited paragraph must also be in the
retrieved list. Cite only the paragraphs that support your answer, not
everything you retrieved. Answer yes/no questions with `Yes` or `No`, and keep
other answers short. The scorer cannot tell whether a quote really justifies
the answer, so a person checks that.

The scorer rejects malformed records, made-up quotes, unknown or repeated IDs,
paragraphs from the wrong paper, and citations outside the retrieved list. A
question with no prediction scores zero. If the model provider fails, the
baseline retries a few times and then stops with an error, and it keeps the
predictions written so far. If the model returns invalid output, the run also
stops, and the baseline does not turn that into an abstention. A rerun
overwrites the output file. Logs go to stderr, and the JSONL file holds only
predictions.

## Application Structure

The installable `qasper_rag` package lives under `src/qasper_rag/`. It is split into folders by role:

```text
src/qasper_rag/
  cli.py              the qasper-rag command and shared command plumbing
  config.py           reads and checks the settings file; loads custom classes
  pipeline/
    service.py        RAGService: retrieve, generate, check; load_service builds it
    retrieval.py      EmbeddingRetriever: cosine search over stored embeddings
    generation.py     ChatGenerator: one chat call that answers with quotes or abstains
  providers/
    api_client.py     authenticated API requests with retries, and TLS setup
    embeddings.py     embedding calls, vector checks, and the SQLite cache
  validation/
    contracts.py      the output rules every prediction must follow
    quotes.py         repairs near-miss citation quotes
  storage/
    jsonl.py          JSON Lines input and output
  commands/
    index.py          qasper-rag index
    predict.py        qasper-rag predict
    answer.py         qasper-rag answer
  web/
    server.py         qasper-rag serve
    static/index.html the chat page
```

New retrievers and generators can live in your own package (see "Extending the Starter") or next to the built-in ones in `pipeline/`.

`predict`, `answer` and `serve` all use `RAGService`, so the scorer measures the same code the chat page runs. The service searches one paper, several papers, or every paper. Evaluation questions name one `paper_id`, so a batch run searches one paper per question. The chat page and `answer` can search every paper, and each citation names its paper. The baseline is simple on purpose and is not expected to score well.

The package uses the [`src` layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
After installation, run `qasper-rag --help` or `python -m qasper_rag --help` to
see the commands.

## Extending the Starter

You can replace any of three parts with your own class without editing the
starter code. The retriever and the generator are the two stages. The service
decides how the stages work together.

| Stage | Built-in class | Your class needs |
| --- | --- | --- |
| Retriever | `qasper_rag.pipeline.retrieval:EmbeddingRetriever` | `__init__(paragraphs, settings, index_dir)` and `retrieve(question, top_k) -> list[paragraph]` |
| Generator | `qasper_rag.pipeline.generation:ChatGenerator` | `__init__(settings)` and `generate(question, passages) -> {answer, abstain, citations}` |
| Service | `qasper_rag.pipeline.service:RAGService` | A subclass that overrides `run(question) -> Draft`, calling `self.retrieve` and `self.generate` |

Name your class in the settings file with `"class": "module:ClassName"`. The
other keys in that section are passed to your class as `settings`, so you can
add your own options:

```json
{
  "top_k": 5,
  "retriever": {
    "class": "my_rag.hybrid:HybridRetriever",
    "embedding": {"model": "openai/text-embedding-3-small", "dimensions": 256},
    "keyword_weight": 0.3
  },
  "generator": {"model": "openai/gpt-6-luna", "structured_output": true}
}
```

Add a `"service": {"class": "..."}` section the same way. Use it for a step
that goes back and forth between the stages, e.g. retrying retrieval when the
evidence is weak, rewriting the question, or checking the answer before you
return it. The other keys in `service` reach your subclass as `self.settings`.
The base class still checks the paper scope and the output rules after `run`,
so a subclass cannot skip them. `Draft` is in `qasper_rag.pipeline.service`.
`self.retrieve` checks the passages, and `self.generate` checks the answer and
regenerates it once if it breaks the rules. The chat server shares one service
object across request threads, so do not store per-question state on `self`.

The module is looked up in the folder you run the command from, then in
installed packages. A `my_rag/` folder in the project root works. A few rules apply to every class:

- `question["paper_ids"]` lists the papers to search. Return passages from those papers only.
- Return the passages unchanged, with their `paragraph_id`, `paper_id` and `text`. The service rejects changed text.
- Return at most `top_k` passages, best first.
- `qasper-rag index` embeds the corpus when the retriever settings have an `embedding` section. Without one, it only saves the corpus, so your retriever can build what it needs from `paragraphs`.
- To build that at index time instead of at every load, give the retriever class a `prepare_index(paragraphs, settings, index_dir)` class method or static method. `qasper-rag index` calls it once after writing the index files, e.g. to embed chunks into a cache in `index_dir` that the retriever then reads when it loads.

A class can wrap another one, e.g. a retriever that calls `EmbeddingRetriever`
for 20 candidates and then reranks them. `tests/test_extension.py` has a small
working retriever, generator and service. The `Retriever` and `Generator`
protocols in `pipeline/service.py` describe the stage interfaces.

The baseline keeps all vectors in memory and searches them in plain Python.
Embedding inputs are cut to 12,000 characters, but the full text is still used
for answers and citations. The chat call requests JSON schema output, and the
service checks IDs and quotes. The starter repairs near-miss quotes and
regenerates an invalid answer once. It has no semantic citation check, query
cache, or resume for an interrupted prediction run. Logs go to stderr.
Predictions written before a failed question are kept, and a rerun overwrites
the file.

| Tool | What it does |
| --- | --- |
| `tools/score.py` | The fixed scorer; propose metric changes separately |
| `tools/run_eval.py` | Runs subset, index, predict and score in one command, with an optional RAGAS step |
| `tools/subset.py` | Copies the first N questions, their gold labels, and only their papers' paragraphs |
| `tools/ragas_eval.py` | Scores answerable questions with RAGAS |
| `tools/download_data.py` | Downloads the QASPER JSON files |
| `tools/prepare_data.py` | Rebuilds `data/` from the downloaded JSON |

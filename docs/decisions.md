# Architecture Decisions

A log of the design choices made on top of the starter, newest at the bottom.
Each entry says what was decided, why, what else was considered, and what it
costs. Status is **Accepted**, **Proposed** (not built yet, open to change) or
**Superseded** (with a link to the entry that replaced it). Evidence for
decisions driven by results lives in [EXPERIMENTS.md](../EXPERIMENTS.md).

---

## D1: Extend the starter through its plug-in points; edit it only deliberately

- **Status:** Accepted, 2026-10-31. Revised the same day: small, deliberate
  starter edits are allowed (they were not allowed at first).
- **Decision:** New strategies live in new modules in `src/qasper_agent/`
  and are turned on by naming classes in a config file
  (`"class": "module:ClassName"`). The starter (`src/qasper_rag/`,
  `pyproject.toml`) is edited only when a plug-in point is missing or a
  workaround would be less clear than a small change. Each edit:
  - is minimal and general (it adds a hook, and does not put our strategy
    logic into the starter);
  - keeps existing behavior and tests passing, with a new test for the
    hook;
  - gets its own commit and decision entry.

  `tools/score.py` and `validation/contracts.py` are never changed (see
  CLAUDE.md).
- **Why:** The starter already provides the indexing, the output contract,
  quote repair, the CLI, the web page and the scorer, and it has tests. Leaving
  it untouched means every experiment is a config diff against a known
  baseline, `predict`, `answer` and `serve` all run our code without changes,
  and the contract checks in `RAGService` still guard every output.
- **Alternatives:**
  - Freely edit `src/qasper_rag/`. Faster for a single change, but it mixes
    our code with reference code and makes it harder to show exactly what
    changed.
  - Never edit it (the first version of this decision). That forced
    awkward workarounds, e.g. embedding at retriever load because
    `qasper-rag index` could only embed raw text (D3).
- **Cost:** Each starter edit needs a short justification and a test.
  Starter edits so far: `pyproject.toml` (D4), the `prepare_index` hook
  in `commands/index.py` (D5).

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
  - **Index text:** what text is indexed for a paragraph, e.g.
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
- **Cost:** The starter's `qasper-rag index` only embeds raw paragraph
  text, so the composed retriever embeds its own units into a cache in
  `index_dir`. At first this happened when the retriever loaded; since D5
  it happens at index time.
- **Interface choices made while building it:**
  - Chunkers split *before* the index text is rendered, so a rule such as
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

## D5: Add an opt-in `prepare_index` hook to `qasper-rag index`

- **Status:** Accepted, 2026-10-31. The first starter code edit under the
  revised D1.
- **Decision:** If the configured retriever class defines
  `prepare_index(paragraphs, settings, index_dir)`, `qasper-rag index`
  calls it once after writing the index files. `ComposedRetriever`
  implements it by building itself, which fits every scorer and fills
  their caches.
- **Why:** Without it, a retriever that needs anything other than raw
  paragraph embeddings had to build at load time. That made the first
  `predict` or `serve` silently slow (minutes for the full corpus, with
  the chat page blocked), made `qasper-rag index` a no-op for our configs,
  and moved API failures from indexing to startup. With the hook,
  indexing is again the one-time slow step, as in the starter.
- **Alternatives:**
  - Keep building at load and add a separate `tools/build_index.py`. No
    starter edit, but a second indexing command for developers to learn.
  - Add a top-level `retriever.embedding` section so the starter embeds
    raw text. That only warms the cache for raw text, not chunks or
    section-prefixed text.
- **Cost:** About 30 lines in `src/qasper_rag/commands/index.py`, a
  paragraph in `docs/baseline.md`, and `tests/test_index_hook.py`.
  Retrievers without the method, including the built-in one, are
  unaffected. The hook runs after the index files are renamed into place,
  so a failure inside it leaves a usable index with a partial cache; the
  retriever finishes the cache on its next load.
- **Evidence:** EXPERIMENTS.md C14: indexing the 20-question subset took
  21 s; loading afterwards took 0.07 s with 0 embedding calls, and
  retrieval matched the baseline on 20/20 questions.

## D6: Evaluate retrieval on 100–150 dev questions, answers on the 20

- **Status:** Accepted, 2026-10-31
- **Decision:** Retrieval-only evaluation (recall@k, no answer model calls)
  runs on 100–150 dev questions generated with
  `tools/prepare_data.py --count`. Full end-to-end runs (retrieval plus
  generation, scored by `tools/score.py`) stay on the shipped 20 unless a
  change needs more evidence.
- **Why:** The shipped sample has 15 answerable questions, so one question
  moves evidence recall by about 7 points (EXPERIMENTS.md 001). Retrieval
  evaluation only embeds the questions (paragraph vectors are cached), so a
  larger set costs almost nothing and makes retrieval comparisons stable.
  End-to-end runs cost one or more chat calls per question and vary from
  run to run (000 vs 001), so they stay small.
- **Alternatives:** Only the 20 (cheapest, but too noisy to rank retrieval
  variants). All ~1,000 dev questions (most stable, but the selection
  filters in `prepare_data.py` would need checking at that size, and the
  extra stability is not needed yet).
- **Cost:** Gold labels for more questions exist on disk, so the rule
  "gold only for scoring" needs care: the evaluation tool reads gold only
  after retrieval has finished.

## D7: Consider a reranker only when retrieval shows headroom

- **Status:** Accepted, 2026-10-31; revised the same day after review. Not
  built, and out of scope for the current iteration.
- **Headroom observed (EXPERIMENTS.md 003, 005):** for the chosen hybrid
  config, recall@5 is 0.65 and recall@20 is 0.92, so headroom exists: 39 of
  62 missed gold paragraphs are ranked 6–20. 23 are below rank 20, out of
  any reranker's reach. A reranker is a candidate to *test* in a later
  iteration, on a subset, against its latency.
- **Decision:** Recall@20 well above recall@5 on the D6 retrieval set is
  *headroom*: the right paragraph is often found but ranked too low. That
  is a necessary sign, not a decision. If headroom appears, a later
  iteration may test a reranker on a representative subset, and keep it
  only if retrieval improves enough to justify its latency and complexity.
  If the gold paragraph is not in the top 20 either, a reranker cannot
  help, and effort goes to scoring and context instead. The `Reranker`
  interface and `candidate_k` already exist in `ComposedRetriever`, so a
  test needs no structural change.
- **Revision:** The first version said to build a reranker whenever the
  gap crossed a fixed threshold. Review pointed out that headroom does not
  show a reranker would close it, so the threshold was dropped.
- **Default if built:** A listwise reranker that asks the existing
  OpenRouter chat model to order about 20 candidates. No new dependencies,
  one extra call (about 1–2 s) per question.
- **Alternatives:** A local cross-encoder (e.g. a BGE reranker): free per
  call and strong, but it needs PyTorch and a model download, a heavy
  addition to a standard-library code base. A hosted rerank API (e.g.
  Cohere, Jina, Voyage): fast, but a second provider and key.
- **Evidence to collect:** recall@5, @10 and @20 for each retriever config
  (see EXPERIMENTS.md once the evaluation tool exists).

## D8: Build the agentic retrieval loop last; considerations so far

- **Status:** Deferred, 2026-10-31. To be built after the retriever and
  generator changes have been measured.
- **Why last:** A loop repeats whatever the retriever and generator do, so
  it multiplies their errors and costs. It is easier to judge once each
  stage is measured on its own.
- **Where it goes:** A `RAGService` subclass named in the `service` config
  section. The starter supports this already (its tests include a
  `RetryOnAbstainService`), and the base class still checks scope and
  output rules after `run`, so no starter change is expected.
- **Triggers suggested by run 001:**
  - The generator abstains, so rewrite the query and retrieve again. 3 of
    the 4 wrong abstentions in 001 were retrieval misses.
  - The answer is not supported by its own quotes, so abstain. 4 of the 5
    missed abstentions in 001 were hedged answers ("the passages do not
    establish…").
- **Evidence threshold:** Use a prompted judgment of sufficiency (enough,
  partial, or not enough, in the style of CRAG's retrieval evaluator), not
  a numeric cutoff on retrieval scores. Raw cosine and BM25 scores are not
  comparable across questions: 0.4 means different things for "What is the
  baseline?" and a long, specific question.
- **Constraints:** The final answer still uses at most 5 retrieved
  paragraphs, and every citation must come from them, so retries must
  choose a final set of 5. Each round adds 1–2 model calls, so latency and
  cost grow 2–3×. Cap at 1–2 retries. Measure on the 20 end-to-end
  questions: abstention precision and recall, and coverage of answerable
  questions.
- **Open questions:** query rewriting versus searching for missing aspects;
  whether the sufficiency check and the answer can share one model call;
  how the loop behaves in all-papers mode.

## D9: Add section-title context to indexed text only if it helps recall

- **Status:** Measured 2026-10-31 (EXPERIMENTS.md 004): **not adopted.**
  `SectionPrefixed` is built and kept as an option, but the default configs
  stay on raw text. With each scorer section titles helped about as many
  questions as it hurt (BM25 3 vs 3, dense 9 vs 8, hybrid 7 vs 6), and
  recall@5 moved by at most +0.01. It fixed the targeted misses ("What
  baseline is used?", "surface-level features"), but a section title is
  shared by every paragraph in the section, so a matching title also lifts
  that section's other paragraphs and pushes evidence elsewhere down.
  LLM-written context (the alternative below) is not pursued for now.
- **Decision:** Test index text that puts the section path in
  front of each paragraph before it is indexed (e.g. "Baseline Approaches:
  We compare our approach to two other…"), and keep it only if recall
  improves on the D6 retrieval set. The paper title is left out for now: it
  only helps in all-papers mode.
- **Why:** The generator already sees each paragraph's section when it
  answers, so context does not matter for citations. It matters for
  *search*: some evidence paragraphs never use the question's words, and
  their section title does. In run 001, Q13 ("What baseline is used?") had
  evidence under "Baseline Approaches", and Q12 ("surface-level features")
  under "Features.". That is 2 of 9 misses, a hypothesis rather than proof.
- **Alternatives:** LLM-written context per paragraph (Anthropic's
  Contextual Retrieval): richer, but about 15.7k LLM calls to index the
  corpus. Only worth trying if the free, deterministic section prefix helps
  and leaves a gap.
- **Cost:** About 15 lines (an `IndexText` class) and one re-index (well under
  one cent of embeddings). BM25 picks it up with no extra cost.

## D10: Parent-child retrieval: diagnose first, no automatic choice

- **Status:** Accepted, 2026-10-31; revised the same day after review.
  Parent-child retrieval is out of scope for the current iteration.
- **Diagnostic notes so far (EXPERIMENTS.md 002–004):** the clearest
  pattern among misses is *list items under one section*. The question asks
  for a set (baselines, systems, strategies), the evidence is several
  sibling paragraphs that each describe one item without repeating the
  question's words, and a sentence announcing the list ranks first. Seen
  in "state of the art" (ELIZA, PARRY, …), "baseline approaches"
  (K-means, LEM, DPEMM), "backoff strategies" and "What are the
  baselines?". This is the strongest argument for testing some
  section-aware expansion later. Single missed paragraphs far from any hit
  are more common and would not be helped by it.
- **Decision:** `tools/retrieval_eval.py` reports, for each question, the
  paragraphs of the best-matching reference that the top 5 missed, with
  their rank in the top 20 and their section, plus the top-20 ranking.
  Those diagnostics, and reading the failures, inform whether any expansion
  is worth testing later. No miss category picks an architecture
  automatically.
- **Revision:** The first version classified misses as adjacent, same
  section or unrelated, and let the largest class choose neighbour
  expansion, section filling or neither. Review pointed out that sharing a
  30-paragraph "Experiments" section, or being adjacent, does not show
  that expansion would help under a 5-paragraph limit. It also counted
  misses over all references, which could report a paragraph as missed
  when another accepted reference was fully retrieved. Diagnostics now use
  only the best-matching reference.
- **Why:** The 5-paragraph limit makes every added context paragraph cost a
  slot. Sections have a median of 2 paragraphs and 82% have at most 5, but
  48% of paragraphs sit in sections with more than 5 (up to 74), so
  "return the whole section" does not always fit. Whether a fixed
  neighbour window is enough is unknown.
- **Rejected:** Showing the model a whole section while reporting only 5
  paragraph IDs. The contract defines `retrieved_paragraph_ids` as the
  paragraphs searched, so the report would understate what the model read.

## D11: Figures and tables: captions only, abstain on their contents

- **Status:** Accepted, 2026-10-31. The prompt line is part of D12.
- **Decision:** No extra code for figures and tables. The 1,929 captions
  (section `Caption`, e.g. "Table 1: Main Results. …") stay searchable and
  citable like any paragraph. The answer prompt says: if the answer depends
  on a table's or figure's contents that only a caption describes, abstain.
  The writeup states this limit.
- **Why:** The corpus has captions but no table cells or images. In-text
  references are placeholders (`TABREF17`) that cannot be matched to caption
  numbers, so mentions cannot be linked to captions. The shipped dev sample
  already excluded questions whose evidence is a figure or table.

## D12: Fix abstention and answer length in the starter prompt

- **Status:** Accepted, 2026-10-31. Revised 2026-11-01: edit the starter's
  prompt in place instead of adding a subclass. Built: abstain instead of
  hedging; short answers and Yes/No; paper names only across papers. Not
  built yet: the figure/table rule.
- **Decision:** Revise `PROMPT` in `src/qasper_rag/pipeline/generation.py`,
  one rule per commit: abstain instead of hedging; answer yes/no questions
  with only "Yes" or "No" and keep other answers short; name paper IDs only
  when more than one paper is in scope; and the figure/table rule from D11.
  No cleanup of the model's output after it answers: the prompt alone has
  to produce the right form. The paper rule is chosen in code by
  `build_prompt`: a question with several papers in scope gets "state which
  paper_id each claim comes from", a one-paper question gets "do not
  mention its paper_id". The starter's single conditional sentence ("If
  passages come from several papers…") was applied to one-paper answers
  too.
- **Why edit the starter:** Our code already relies on the starter for
  everything except retrieval (generator, service, output checks, CLI), and
  the prompt is one constant. A subclass would be a second generator that
  differs only by that string. The starter's original prompt stays in git
  history (commit `f67a7d2` and earlier), which is what runs 000–005 used.
- **What the brief asks for:** abstain "even if it could guess", using one
  of our hedged questions as its example (ASSIGNMENT.md, "Abstaining");
  "Answer yes/no questions with `Yes` or `No`, and keep other answers
  short" (docs/data.md, docs/baseline.md); name the paper for each claim
  when the question spans papers (ASSIGNMENT.md, "Across papers").
- **Why:** Run 001: 4 of 5 unanswerable questions got hedged answers ("the
  passages do not establish…") instead of abstentions, and the prompt's
  "state which paper_id each claim comes from" made single-paper answers
  wordy ("Yes. Paper 1705.01265 reports…" where the gold is "Yes"), which
  lowers token F1 even when the answer is right.
- **Measurement:** End-to-end runs: abstention precision and recall and
  coverage first, then answer token F1 read next to the answers themselves,
  because token F1 understates answers that are right but long or worded
  differently (docs/evaluation.md). Answers vary between runs at
  temperature 0 (000 vs 001), so small changes need care.

## D13: Hold off on query rewriting and HyDE

- **Status:** Deferred, 2026-10-31
- **Decision:** No query rewriting (an LLM expands the question before
  search) and no HyDE (an LLM writes a hypothetical answer paragraph that is
  embedded instead of the question) for now.
- **Why:** Both target short, generic questions such as "What is the
  baseline?", which BM25 (exact words) and section context (D9) also target,
  at no per-query cost. Both add one LLM call (about 1–2 s) per question.
- **Revisit if:** After BM25, hybrid and D9, misses remain on questions whose
  wording differs from the paper's. Query rewriting is also a candidate
  retry action for the agentic loop (D8).

## D14: Cache question embeddings in `DenseScorer`

- **Status:** Accepted, 2026-10-31
- **Decision:** `DenseScorer.score` embeds the question with the starter's
  `EmbeddingClient.embed_cached`, in the same cache as the paragraphs,
  instead of calling the API every time.
- **Why:** EXPERIMENTS.md 002 found that the embedding API returns
  slightly different vectors for the same text between calls (50 of 120
  questions differed, up to 5.4e-3 per component), so about 1 question in
  120 could swap a paragraph at the top-5 boundary between runs. Caching
  freezes one vector per question: reruns become exactly repeatable, and
  dense and hybrid configs are compared on identical question vectors. It
  also saves API calls for repeated questions.
- **Alternatives:** Leave it uncached and treat differences of about 0.01
  recall as noise. Simpler, but comparisons of close configs would be
  harder to trust.
- **Cost:** One SQLite read per question (about a millisecond). Each new
  question adds a cache entry; the key is a hash of the model settings and
  the text, so the question text itself is not stored. The cache freezes
  the first vector it sees, so results are repeatable but still reflect
  one sample of the API's variation. The starter's `EmbeddingRetriever` is
  unchanged.
- **Evidence:** EXPERIMENTS.md 002b (the new dense reference).

## D15: Default retriever: hybrid BM25 + dense

- **Status:** Accepted, 2026-10-31
- **Decision:** `config/composed-hybrid.json` (raw text, whole paragraphs,
  BM25 + dense `text-embedding-3-small` at 256 dimensions, reciprocal rank
  fusion with k 60, no credit for non-matches) is the agent's default
  retriever. The starter's `config/baseline.json` stays as the reference
  baseline.
- **Why:** On 100 answerable dev questions it has the best recall@5 (0.65
  vs 0.59 dense and 0.54 BM25), is never worse than both single scorers,
  and fixes the exact-word misses dense had (EXPERIMENTS.md 003). End to
  end on the shipped 20 it doubles evidence recall@5 over run 001 (0.33 →
  0.63) and cuts retrieval misses from 9 to 4 (005). Section context (D9)
  added nothing.
- **Strength of the evidence:** the recall@5 lead over dense rests on 15
  wins against 9 losses (sign test p = 0.31), so it is suggestive, not
  conclusive. Hybrid's losses have a known cause (BM25's weak matches on
  function words), with stopword removal as the next candidate.
- **Cost:** No extra API calls per question beyond dense (BM25 is local;
  question embeddings are cached, D14). Index build is the same as dense.

## D16: BM25 query stopwords: measured, not adopted

- **Status:** Measured 2026-11-01 (EXPERIMENTS.md 007): **not adopted.**
- **Decision:** Keep the `stopwords` setting on `BM25Scorer`, off by
  default. The default hybrid config does not use it.
- **Why:** 003 traced most hybrid losses to function-word matches, so
  removing stopwords from the query was the cheapest fix. Measured on 100
  answerable questions it moved hybrid recall@5 from 0.65 to 0.64 (3 wins,
  4 losses) and BM25 alone from 0.54 to 0.50. It fixed 2 of the 9 hybrid
  losses against dense but broke 4 other questions, where function words
  had been breaking ties between paragraphs that share the same generic
  content words.
- **Scope:** Query side only, so paragraph lengths and term rarity are
  unchanged and the result is attributable to the stopwords alone. The
  plural rule was not changed in the same experiment (it has its own known
  flaw, noted in EXPERIMENTS.md 003).
- **Revised 2026-11-01:** "not" and "no" were removed from the list, since
  negation can reverse a question's meaning. Retrieval was unchanged on
  the dev set (EXPERIMENTS.md 007b: same recall at every depth), which has
  only 3 questions with "not" or "no". Still off by default.

## D17: No per-paper cap for cross-paper retrieval

- **Status:** Measured 2026-11-01 (EXPERIMENTS.md 008): **not adopted.**
- **Decision:** Keep the plain top 5 across all papers. No per-paper limit
  in `ComposedRetriever`.
- **Why:** On 10 broad cross-paper questions, at most 2 passages per paper
  added relevant papers to 4 questions (fake news 1 → 3 papers; XNLI,
  ROUGE, ELMo +1 each) but added irrelevant evidence to 2 narrow ones
  (sarcasm, CrowdFlower). Every dropped passage was relevant. The cap also
  applies to every multi-paper query, including specific questions asked
  without a paper ID, where it would force in unrelated papers; separating
  the two needs query routing, which is out of scope.
- **Limitation recorded:** across all papers the answer sees 5 passages,
  so "which papers" answers are partial and one paper can crowd out
  others (EXPERIMENTS.md 006, 008).
- **Revisit if:** a larger cross-paper set, judged by someone other than
  the builder, shows a consistent gain, or list-style questions can be
  recognized cheaply.

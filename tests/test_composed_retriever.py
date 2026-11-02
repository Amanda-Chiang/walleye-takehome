import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qasper_agent.components import build_component
from qasper_agent.retrieval.base import Unit
from qasper_agent.retrieval.dense import DenseScorer
from qasper_agent.retrieval.composed import ComposedRetriever, best_paragraphs
from qasper_agent.retrieval.fusion import ReciprocalRankFusion
from qasper_agent.retrieval.index_text import SectionPrefixed
from qasper_rag.commands.index import build_index
from qasper_rag.config import ConfigurationError
from qasper_rag.pipeline.retrieval import EmbeddingRetriever
from qasper_rag.pipeline.service import load_service
from qasper_rag.providers.embeddings import EmbeddingClient, normalize
from qasper_rag.storage.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/fixtures/corpus.jsonl"
ENV = {"RAG_API_KEY": "fixture", "RAG_EMBEDDING_MODEL": "fixture-embedding"}
EMBEDDING = {"model": "fixture-embedding", "dimensions": 4}
PARAGRAPHS = [
    {"paragraph_id": "a:p0", "paper_id": "a", "section": "Intro", "text": "Cats sleep. Dogs bark."},
    {"paragraph_id": "a:p1", "paper_id": "a", "section": "Results", "text": "Dogs bark loudly at dogs."},
    {"paragraph_id": "b:p0", "paper_id": "b", "section": "Intro", "text": "Dogs bark everywhere."},
]


class KeywordScorer:
    """Counts question words in a unit, with no API call."""

    def __init__(self, settings, index_dir):
        self.units = []

    def fit(self, units):
        self.units = units

    def score(self, query, candidates):
        words = set(query.lower().split())
        return [sum(w in words for w in self.units[i].text.lower().split()) for i in candidates]


class SentenceChunker:
    """Splits a paragraph into sentences."""

    def __init__(self, settings):
        pass

    def split(self, paragraph):
        return [s for s in paragraph["text"].split(". ") if s]


class ReverseReranker:
    def __init__(self, settings):
        pass

    def rerank(self, question, passages):
        return list(reversed(passages))


def fake_embed(self, texts):
    """A deterministic, normalized embedding that depends on the text.

    Normalized like the real ``EmbeddingClient.embed``.
    """
    return [normalize([1.0 + len(t) % 3, t.count("a"), t.count("e"), t.count("o")], 4)
            for t in texts]


def keyword_settings(**extra):
    return {"scorers": [{"class": "test_composed_retriever:KeywordScorer"}], **extra}


class ComposedRetrieverTests(unittest.TestCase):
    def test_searches_only_requested_papers_and_returns_unchanged_paragraphs(self):
        retriever = ComposedRetriever(PARAGRAPHS, keyword_settings(), "unused")
        passages = retriever.retrieve({"question": "dogs bark", "paper_ids": ["a"]}, 5)
        self.assertEqual([p["paragraph_id"] for p in passages], ["a:p1", "a:p0"])
        self.assertIs(passages[0], PARAGRAPHS[1])
        self.assertEqual(retriever.retrieve({"question": "dogs", "paper_ids": ["none"]}, 5), [])

    def test_returns_at_most_top_k(self):
        retriever = ComposedRetriever(PARAGRAPHS, keyword_settings(), "unused")
        passages = retriever.retrieve({"question": "dogs", "paper_ids": ["a", "b"]}, 1)
        self.assertEqual(len(passages), 1)

    def test_chunk_scores_roll_up_to_the_best_unit_per_paragraph(self):
        settings = keyword_settings(chunker={"class": "test_composed_retriever:SentenceChunker"})
        retriever = ComposedRetriever(PARAGRAPHS, settings, "unused")
        self.assertEqual(len(retriever.units), 4)
        passages = retriever.retrieve({"question": "cats sleep", "paper_ids": ["a"]}, 5)
        self.assertEqual([p["paragraph_id"] for p in passages], ["a:p0", "a:p1"])

    def test_reranker_sees_candidate_k_paragraphs(self):
        settings = keyword_settings(
            reranker={"class": "test_composed_retriever:ReverseReranker"}, candidate_k=2)
        retriever = ComposedRetriever(PARAGRAPHS, settings, "unused")
        passages = retriever.retrieve({"question": "dogs bark", "paper_ids": ["a"]}, 1)
        self.assertEqual([p["paragraph_id"] for p in passages], ["a:p0"])

    def test_missing_scorers_is_a_clear_error(self):
        with self.assertRaises(ConfigurationError):
            ComposedRetriever(PARAGRAPHS, {}, "unused")
        with self.assertRaises(ConfigurationError):
            ComposedRetriever(PARAGRAPHS, {"scorers": [{}]}, "unused")


class PartTests(unittest.TestCase):
    def test_rrf_keeps_a_single_scorers_order_and_rewards_agreement(self):
        fusion = ReciprocalRankFusion({"k": 1})
        self.assertEqual(fusion.fuse([[0.1, 0.9, 0.5]]), [1 / 4, 1 / 2, 1 / 3])
        fused = fusion.fuse([[3, 2, 1], [1, 3, 2]])
        self.assertEqual(max(range(3), key=fused.__getitem__), 1)

    def test_best_paragraphs_takes_max_and_breaks_ties_by_position(self):
        self.assertEqual(best_paragraphs(["x", "y", "x", "z"], [1, 2, 3, 2]), ["x", "y", "z"])
        # Ties follow first appearance, not ID order.
        self.assertEqual(best_paragraphs(["z", "y", "z"], [1, 1, 0]), ["z", "y"])

    def test_build_component_uses_default_or_named_class(self):
        self.assertIsInstance(build_component("fusion", None, ReciprocalRankFusion), ReciprocalRankFusion)
        with self.assertRaises(ConfigurationError):
            build_component("scorer", {}, None)
        with self.assertRaises(ConfigurationError):
            build_component("scorer", "not a dict", None)


class DenseQueryCacheTests(unittest.TestCase):
    def test_repeated_questions_reuse_their_cached_embedding(self):
        # The embedding API returns slightly different vectors between calls
        # (EXPERIMENTS.md 002), so a cached question vector keeps reruns and
        # config comparisons on identical inputs.
        settings = {"class": "qasper_agent.retrieval.dense:DenseScorer", "embedding": EMBEDDING}
        units = [Unit("x:p0", "x", "alpha text"), Unit("x:p1", "x", "beta words")]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV), \
                patch.object(EmbeddingClient, "embed", autospec=True,
                             side_effect=fake_embed) as embed:
            scorer = DenseScorer(settings, tmp)
            scorer.fit(units)
            calls_after_fit = embed.call_count
            first = scorer.score("same question", [0, 1])
            self.assertEqual(scorer.score("same question", [0, 1]), first)
            self.assertEqual(embed.call_count, calls_after_fit + 1)
            # A new scorer (e.g. a later evaluation run) reads the cache too.
            reloaded = DenseScorer(settings, tmp)
            reloaded.fit(units)
            reloaded.score("same question", [0, 1])
            self.assertEqual(embed.call_count, calls_after_fit + 1)
            scorer.score("another question", [0, 1])
            self.assertEqual(embed.call_count, calls_after_fit + 2)


class SectionIndexTextTests(unittest.TestCase):
    def test_puts_the_section_path_before_the_text(self):
        paragraph = {"section": "Results ::: Parsing", "text": "LAS is 80."}
        self.assertEqual(SectionPrefixed({}).render(paragraph, "LAS is 80."),
                         "Results > Parsing\nLAS is 80.")

    def test_empty_section_leaves_the_text_unchanged(self):
        self.assertEqual(SectionPrefixed({}).render({"section": "", "text": "t"}, "t"), "t")

    def test_section_words_become_searchable(self):
        # EXPERIMENTS.md 001/002: the evidence never says "baseline"; its section does.
        paragraphs = [
            {"paragraph_id": "x:p00000", "paper_id": "x", "section": "Introduction",
             "text": "We study emotion detection and compare our approach to others."},
            {"paragraph_id": "x:p00001", "paper_id": "x", "section": "Baseline Approaches",
             "text": "We compare our approach to two other systems."},
        ]
        bm25 = [{"class": "qasper_agent.retrieval.bm25:BM25Scorer"}]
        index_text = {"class": "qasper_agent.retrieval.index_text:SectionPrefixed"}
        question = {"question": "What baseline is used?", "paper_ids": ["x"]}
        raw = ComposedRetriever(paragraphs, {"scorers": bm25}, "unused")
        section = ComposedRetriever(paragraphs, {"scorers": bm25, "index_text": index_text}, "unused")
        self.assertEqual(raw.retrieve(question, 1)[0]["paragraph_id"], "x:p00000")
        self.assertIs(section.retrieve(question, 1)[0], paragraphs[1])


class BaselineParityTests(unittest.TestCase):
    """Raw text + whole paragraphs + dense scoring must rank like the starter."""

    def test_dense_composition_matches_embedding_retriever(self):
        composed_settings = {"scorers": [{"class": "qasper_agent.retrieval.dense:DenseScorer",
                                          "embedding": EMBEDDING}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV), \
                patch.object(EmbeddingClient, "embed", fake_embed):
            config = Path(tmp) / "baseline.json"
            config.write_text(json.dumps({"retriever": {"embedding": EMBEDDING}, "generator": {}}))
            build_index(CORPUS, tmp, config)
            paragraphs = read_jsonl(CORPUS)
            baseline = EmbeddingRetriever(paragraphs, {"embedding": EMBEDDING}, tmp)
            composed = ComposedRetriever(paragraphs, composed_settings, tmp)
            for text in ("What accuracy?", "classifier", "another set"):
                question = {"question": text, "paper_ids": ["demo", "other"]}
                with self.subTest(text=text):
                    self.assertEqual(
                        [p["paragraph_id"] for p in composed.retrieve(question, 3)],
                        [p["paragraph_id"] for p in baseline.retrieve(question, 3)])

    def test_index_command_fills_the_cache_so_loading_embeds_nothing(self):
        config = {"retriever": {"class": "qasper_agent.retrieval.composed:ComposedRetriever",
                                "scorers": [{"class": "qasper_agent.retrieval.dense:DenseScorer",
                                             "embedding": EMBEDDING}]},
                  "generator": {}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV), \
                patch.object(EmbeddingClient, "embed", autospec=True,
                             side_effect=fake_embed) as embed:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            build_index(CORPUS, tmp, path)
            self.assertTrue((Path(tmp) / "embedding-cache.sqlite").exists())
            calls_at_index = embed.call_count
            ComposedRetriever(read_jsonl(CORPUS), config["retriever"], tmp)
        self.assertGreater(calls_at_index, 0)
        self.assertEqual(embed.call_count, calls_at_index)

    def test_composed_retriever_loads_through_the_starter_service(self):
        config = {"retriever": {"class": "qasper_agent.retrieval.composed:ComposedRetriever",
                                "scorers": [{"class": "test_composed_retriever:KeywordScorer"}]},
                  "generator": {"class": "test_extension:QuoteGenerator"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            build_index(CORPUS, tmp, path)
            service = load_service(tmp, path)
            response = service.ask("accuracy evaluation", paper_ids=["demo"])
        self.assertIsInstance(service.retriever, ComposedRetriever)
        self.assertEqual(response["retrieved_paragraph_ids"][0], "demo:p00000")


if __name__ == "__main__":
    unittest.main()

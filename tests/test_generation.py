import json
import os
import unittest
from unittest.mock import patch

from qasper_rag.pipeline.generation import (
    ONE_PAPER_RULE,
    SEVERAL_PAPERS_RULE,
    ChatGenerator,
    build_prompt,
)

PASSAGES = [
    {"paper_id": "a", "paragraph_id": "a:p00001", "text": "We use English tweets."},
    {"paper_id": "b", "paragraph_id": "b:p00002", "text": "We use German news."},
]


class PaperRuleTests(unittest.TestCase):
    def test_one_paper_in_scope_says_not_to_name_it(self):
        prompt = build_prompt({"question": "q", "paper_ids": ["a"]}, PASSAGES[:1])
        self.assertIn(ONE_PAPER_RULE, prompt)
        self.assertNotIn(SEVERAL_PAPERS_RULE, prompt)

    def test_several_papers_in_scope_name_each_claim(self):
        # Scope decides, even if the passages all come from one paper.
        prompt = build_prompt({"question": "q", "paper_ids": ["a", "b"]}, PASSAGES[:1])
        self.assertIn(SEVERAL_PAPERS_RULE, prompt)
        self.assertNotIn(ONE_PAPER_RULE, prompt)

    def test_without_scope_the_passages_papers_decide(self):
        self.assertIn(SEVERAL_PAPERS_RULE, build_prompt({"question": "q"}, PASSAGES))
        self.assertIn(ONE_PAPER_RULE, build_prompt({"question": "q"}, PASSAGES[:1]))

    def test_question_and_passages_follow_the_instructions_as_json(self):
        prompt = build_prompt({"question": "Which data?", "paper_ids": ["a"]}, PASSAGES[:1])
        data = json.loads(prompt.split(ONE_PAPER_RULE, 1)[1])
        self.assertEqual(data, {"question": "Which data?", "passages": PASSAGES[:1]})

    def test_generator_sends_the_built_prompt(self):
        env = {"RAG_BASE_URL": "http://localhost/v1", "RAG_MODEL": "fake", "RAG_API_KEY": "test"}
        with patch.dict(os.environ, env):
            generator = ChatGenerator({})
        answer = {"answer": "Yes", "abstain": False, "citations": []}
        reply = {"choices": [{"message": {"content": json.dumps(answer)}}]}
        question = {"question": "q", "paper_ids": ["a"]}
        with patch.object(generator.client, "post", return_value=reply) as post:
            self.assertEqual(generator.generate(question, PASSAGES[:1]), answer)
        sent = post.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(sent, build_prompt(question, PASSAGES[:1]))


if __name__ == "__main__":
    unittest.main()

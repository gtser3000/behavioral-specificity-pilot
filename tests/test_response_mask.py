"""Loss-mask checks against the actual downloaded Qwen tokenizer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark import encode_record
from transformers import AutoTokenizer


class ResponseMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / '.cache/model', local_files_only=True)

    def test_only_assistant_answer_is_supervised(self):
        messages = [{'role': 'user', 'content': 'What is two plus two?'},
                    {'role': 'assistant', 'content': 'Four.'}]
        row = encode_record(self.tokenizer, {'messages': messages}, 512)
        prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        self.assertEqual(len(row['labels']), len(row['input_ids']))
        self.assertTrue(all(x == -100 for x in row['labels'][:len(prompt)]))
        self.assertEqual(row['labels'][len(prompt):], row['input_ids'][len(prompt):])
        supervised = self.tokenizer.decode([x for x in row['labels'] if x != -100])
        self.assertIn('Four.', supervised)
        self.assertNotIn('two plus two', supervised)

    def test_context_limit_excluding_answer_rejects_record(self):
        messages = [{'role': 'user', 'content': 'Please answer this question.'},
                    {'role': 'assistant', 'content': 'Here is an answer.'}]
        prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        self.assertIsNone(encode_record(self.tokenizer, {'messages': messages}, len(prompt)))

    def test_multiple_assistant_answers_are_rejected(self):
        row = {'messages': [{'role': 'user', 'content': 'Hello'},
                            {'role': 'assistant', 'content': 'Hello'},
                            {'role': 'user', 'content': 'What is two plus two?'},
                            {'role': 'assistant', 'content': 'Four'}]}
        self.assertIsNone(encode_record(self.tokenizer, row, 512))


if __name__ == '__main__':
    unittest.main()

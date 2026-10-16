import unittest

from minigpt.data.sft_format import build_prompt, parse_example
from minigpt.tokenizer import BPETokenizer

HEADER_FIRST = (
    "Features: Dialogue\n"
    "Words: quit, oak, gloomy\n"
    "Summary: Sara and Ben played in the park.\n"
    "Story: \n\n"
    "Sara and Ben were playing.\n"
    "But the sky was gloomy.\n"
)

SUMMARY_AFTER = (
    "Words: help, mud, bossy\n"
    "Story: \n\n"
    "Lily and Sam were playing.\n"
    "They became friends.\n"
    "Summary: Lily and Sam play in the mud.\n"
)

SINGLE_NEWLINE = (
    "Words: crawl, card\n"
    "Random sentence: They say thank you.\n"
    "Story: \n"
    "Lily and Ben are twins.\n"
)


class TestParseExample(unittest.TestCase):
    def test_header_before_story(self):
        fields, story = parse_example(HEADER_FIRST)
        self.assertEqual(
            fields,
            {
                "Features": "Dialogue",
                "Words": "quit, oak, gloomy",
                "Summary": "Sara and Ben played in the park.",
            },
        )
        self.assertEqual(story, "Sara and Ben were playing.\nBut the sky was gloomy.")

    def test_summary_after_story_moves_to_fields(self):
        fields, story = parse_example(SUMMARY_AFTER)
        self.assertEqual(fields["Summary"], "Lily and Sam play in the mud.")
        self.assertEqual(fields["Words"], "help, mud, bossy")
        self.assertEqual(story, "Lily and Sam were playing.\nThey became friends.")

    def test_single_newline_after_story_marker(self):
        fields, story = parse_example(SINGLE_NEWLINE)
        self.assertEqual(fields["Random sentence"], "They say thank you.")
        self.assertEqual(story, "Lily and Ben are twins.")

    def test_unknown_header_line_is_rejected(self):
        self.assertIsNone(parse_example("Words: a\nMystery: x\nStory:\nHello there."))

    def test_missing_story_marker_is_rejected(self):
        self.assertIsNone(parse_example("Words: a, b\nHello there."))

    def test_empty_story_is_rejected(self):
        self.assertIsNone(parse_example("Words: a, b\nStory: "))


class TestBuildPrompt(unittest.TestCase):
    def test_canonical_field_order(self):
        fields = {"Words": "a, b", "Features": "Dialogue", "Summary": "A summary."}
        self.assertEqual(
            build_prompt(fields),
            "Features: Dialogue\nWords: a, b\nSummary: A summary.\nStory:\n",
        )

    def test_no_fields(self):
        self.assertEqual(build_prompt({}), "Story:\n")

    def test_parse_then_build_is_canonical(self):
        fields, _ = parse_example(SUMMARY_AFTER)
        self.assertEqual(
            build_prompt(fields),
            "Words: help, mud, bossy\nSummary: Lily and Sam play in the mud.\nStory:\n",
        )


class TestTokenizationBoundary(unittest.TestCase):
    def test_prompt_and_story_tokenize_independently(self):
        corpus = [HEADER_FIRST, SUMMARY_AFTER, SINGLE_NEWLINE] * 20
        tokenizer = BPETokenizer.train(corpus, vocab_size=330)
        for document in (HEADER_FIRST, SUMMARY_AFTER, SINGLE_NEWLINE):
            fields, story = parse_example(document)
            prompt = build_prompt(fields)
            self.assertEqual(
                tokenizer.encode(prompt) + tokenizer.encode(story),
                tokenizer.encode(prompt + story),
            )


if __name__ == "__main__":
    unittest.main()

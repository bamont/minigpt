import random
import unittest

from minigpt.data.preference_pairs import Sample, select_pair
from minigpt.data.scoring import required_words, score_story


class TestScoring(unittest.TestCase):
    def test_required_words_are_normalized(self):
        self.assertEqual(required_words("Dragon, cake ,moon"), ["dragon", "cake", "moon"])

    def test_exact_and_prefix_scores(self):
        exact, prefix = score_story("The dragons baked a cake.", ["dragon", "cake", "moon"])
        self.assertEqual(exact, 1)
        self.assertEqual(prefix, 2)

    def test_case_insensitive(self):
        exact, _ = score_story("The MOON was bright.", ["moon"])
        self.assertEqual(exact, 1)


def sample(score: int, length: int = 100, finished: bool = True, logprob: float = -1.0) -> Sample:
    return Sample(ids=list(range(length)), finished=finished, score=score, logprob=logprob)


class TestSelectPair(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)

    def test_picks_highest_finished_and_lowest_scoring(self):
        samples = [sample(1), sample(3), sample(0), sample(2)]
        chosen, rejected = select_pair(samples, self.rng)
        self.assertEqual((chosen.score, rejected.score), (3, 0))

    def test_no_pair_when_scores_are_equal(self):
        self.assertIsNone(select_pair([sample(2), sample(2), sample(2)], self.rng))

    def test_no_pair_without_finished_sample(self):
        samples = [sample(3, finished=False), sample(1, finished=False)]
        self.assertIsNone(select_pair(samples, self.rng))

    def test_unfinished_sample_is_never_chosen(self):
        samples = [sample(3, finished=False), sample(2), sample(0)]
        chosen, rejected = select_pair(samples, self.rng)
        self.assertEqual((chosen.score, rejected.score), (2, 0))

    def test_unfinished_sample_can_be_rejected(self):
        samples = [sample(2), sample(0, finished=False)]
        chosen, rejected = select_pair(samples, self.rng)
        self.assertFalse(rejected.finished)

    def test_length_ratio_filters_rejected_candidates(self):
        samples = [sample(3, length=100), sample(0, length=450), sample(2, length=120)]
        chosen, rejected = select_pair(samples, self.rng, max_length_ratio=2.0)
        self.assertEqual((chosen.score, rejected.score), (3, 2))

    def test_no_pair_when_every_candidate_is_too_different_in_length(self):
        samples = [sample(3, length=100), sample(0, length=450)]
        self.assertIsNone(select_pair(samples, self.rng, max_length_ratio=2.0))

    def test_selection_is_deterministic_for_a_seed(self):
        samples = [sample(3), sample(3), sample(0), sample(0)]
        first = select_pair(samples, random.Random(5))
        second = select_pair(samples, random.Random(5))
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_unnatural_best_sample_is_not_chosen(self):
        samples = [
            sample(3, logprob=-2.0),
            sample(2, logprob=-0.9),
            sample(0, logprob=-1.0),
            sample(1, logprob=-1.1),
        ]
        chosen, rejected = select_pair(samples, self.rng)
        self.assertEqual((chosen.score, rejected.score), (2, 0))

    def test_rejected_much_more_natural_than_chosen_is_skipped(self):
        samples = [
            sample(2, logprob=-0.6),
            sample(1, logprob=-1.0),
            sample(0, logprob=-0.4),
            sample(0, logprob=-1.2),
        ]
        chosen, rejected = select_pair(samples, self.rng, logprob_tolerance=0.05)
        self.assertEqual(chosen.score, 2)
        self.assertEqual((rejected.score, rejected.logprob), (0, -1.2))

    def test_tolerance_allows_slightly_more_natural_rejected(self):
        samples = [sample(2, logprob=-0.6), sample(0, logprob=-0.58), sample(1, logprob=-1.5)]
        chosen, rejected = select_pair(samples, self.rng, logprob_tolerance=0.05)
        self.assertEqual((chosen.score, rejected.score), (2, 0))


if __name__ == "__main__":
    unittest.main()

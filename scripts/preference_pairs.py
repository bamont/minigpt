import random
import statistics
from dataclasses import dataclass


@dataclass
class Sample:
    ids: list[int]
    finished: bool
    score: int
    logprob: float = 0.0


def select_pair(
    samples: list[Sample],
    rng: random.Random,
    max_length_ratio: float = 2.0,
    logprob_tolerance: float = 0.05,
) -> tuple[Sample, Sample] | None:
    usable = [s for s in samples if s.ids]
    if not usable:
        return None
    median_logprob = statistics.median(s.logprob for s in usable)

    eligible = [s for s in usable if s.finished and s.logprob >= median_logprob]
    if not eligible:
        return None
    best_score = max(s.score for s in eligible)
    chosen = rng.choice([s for s in eligible if s.score == best_score])

    candidates = [
        s
        for s in usable
        if s.score < chosen.score
        and 1 / max_length_ratio <= len(chosen.ids) / len(s.ids) <= max_length_ratio
        and chosen.logprob >= s.logprob - logprob_tolerance
    ]
    if not candidates:
        return None
    worst_score = min(s.score for s in candidates)
    rejected = rng.choice([s for s in candidates if s.score == worst_score])
    return chosen, rejected

import random
from dataclasses import dataclass


@dataclass
class Sample:
    ids: list[int]
    finished: bool
    score: int


def select_pair(
    samples: list[Sample], rng: random.Random, max_length_ratio: float = 2.0
) -> tuple[Sample, Sample] | None:
    finished = [s for s in samples if s.finished and s.ids]
    if not finished:
        return None
    best_score = max(s.score for s in finished)
    chosen = rng.choice([s for s in finished if s.score == best_score])

    candidates = [
        s
        for s in samples
        if s.ids
        and s.score < chosen.score
        and 1 / max_length_ratio <= len(chosen.ids) / len(s.ids) <= max_length_ratio
    ]
    if not candidates:
        return None
    worst_score = min(s.score for s in candidates)
    rejected = rng.choice([s for s in candidates if s.score == worst_score])
    return chosen, rejected

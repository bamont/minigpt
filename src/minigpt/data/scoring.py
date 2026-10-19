import re

WORD_PATTERN = re.compile(r"[a-z']+")


def required_words(words_field: str) -> list[str]:
    return [word.strip().lower() for word in words_field.split(",") if word.strip()]


def score_story(story: str, words: list[str]) -> tuple[int, int]:
    vocabulary = set(WORD_PATTERN.findall(story.lower()))
    exact = sum(word in vocabulary for word in words)
    prefix = sum(any(token.startswith(word) for token in vocabulary) for word in words)
    return exact, prefix

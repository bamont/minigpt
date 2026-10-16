FIELD_NAMES = ("Features", "Words", "Summary", "Random sentence")
STORY_MARKER = "Story:"


def _split_field(line: str) -> tuple[str, str] | None:
    for name in FIELD_NAMES:
        prefix = f"{name}:"
        if line.startswith(prefix):
            return name, line[len(prefix):].strip()
    return None


def parse_example(document: str) -> tuple[dict[str, str], str] | None:
    fields: dict[str, str] = {}
    story_lines: list[str] = []
    in_story = False

    for line in document.strip().splitlines():
        if in_story:
            story_lines.append(line)
        elif line.startswith(STORY_MARKER):
            in_story = True
            remainder = line[len(STORY_MARKER):].strip()
            if remainder:
                story_lines.append(remainder)
        elif line.strip():
            parsed = _split_field(line)
            if parsed is None:
                return None
            fields[parsed[0]] = parsed[1]

    while story_lines:
        tail = story_lines[-1]
        if tail.strip():
            parsed = _split_field(tail)
            if parsed is None:
                break
            fields.setdefault(parsed[0], parsed[1])
        story_lines.pop()

    story = "\n".join(story_lines).strip()
    if not in_story or not story:
        return None
    return fields, story


def build_prompt(fields: dict[str, str]) -> str:
    header = "".join(f"{name}: {fields[name]}\n" for name in FIELD_NAMES if fields.get(name))
    return f"{header}{STORY_MARKER}\n"

import re


_SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_BULLET_RE = re.compile(r"^[\s]*[•·●◦▪▫–—\-]+[\s]+", re.MULTILINE)
_DASHES_RE = re.compile(r"[‐‑‒–—−]")


def clean_text(raw: str) -> str:
    if not raw:
        return ""

    text = raw
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n")
    text = _DASHES_RE.sub("-", text)

    text = _BULLET_RE.sub("- ", text)

    lines = [line.strip() for line in text.split("\n")]
    compact_lines = []
    prev_empty = False
    for line in lines:
        if line:
            compact_lines.append(line)
            prev_empty = False
            continue
        if not prev_empty:
            compact_lines.append("")
            prev_empty = True
    text = "\n".join(compact_lines)

    text = _SPACE_RE.sub(" ", text)

    text = text.strip()

    return text

import re


_SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
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
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    text = _SPACE_RE.sub(" ", text)

    text = _NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    return text

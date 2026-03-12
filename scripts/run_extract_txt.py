from __future__ import annotations

import json
import sys
from pathlib import Path

from schemas.models import ExtractionInput
from services.extract_service import extract_structured_resume
from utils.errors import AppError


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_extract_txt.py <path_to_txt>")
        return 1

    txt_path = Path(sys.argv[1])

    if not txt_path.exists():
        print(f"ERROR: TXT file not found: {txt_path}")
        return 2

    try:
        text = txt_path.read_text(encoding="utf-8")

        extraction_input = ExtractionInput(
            resume_id=txt_path.stem,
            text=text,
        )

        result = extract_structured_resume(extraction_input)

    except AppError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}")
        return 3

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
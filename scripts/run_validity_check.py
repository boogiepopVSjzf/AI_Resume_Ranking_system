from __future__ import annotations

import json
import sys
from pathlib import Path

from utils.resume_validity_checker import ResumeValidityChecker


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_validity_check.py <path_to_txt>")
        return 1

    txt_path = Path(sys.argv[1])

    if not txt_path.exists():
        print(f"ERROR: File not found: {txt_path}")
        return 2

    checker = ResumeValidityChecker()
    result = checker.check_file(txt_path)

    output = {
        "decision": result.decision,
        "confidence": result.confidence,
        "overall_score": result.overall_score,
        "component_scores": result.component_scores,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "stats": {
            "char_count": result.stats.char_count,
            "word_count": result.stats.word_count,
            "line_count": result.stats.line_count,
            "non_empty_line_count": result.stats.non_empty_line_count,
            "section_hits": result.stats.section_hits,
            "date_matches": result.stats.date_matches,
            "email_found": result.stats.email_found,
            "phone_found": result.stats.phone_found,
            "linkedin_found": result.stats.linkedin_found,
            "github_found": result.stats.github_found,
            "bullet_lines": result.stats.bullet_lines,
            "repeated_line_ratio": result.stats.repeated_line_ratio,
            "possible_jd_signals": result.stats.possible_jd_signals,
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
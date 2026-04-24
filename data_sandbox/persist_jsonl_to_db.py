"""Load sandbox JSONL (raw_json per line) through the same bundle + DB path as production.

Pipeline per row:
  dict → ResumeStructured.model_validate
  → build_resume_storage_bundle (metadata, semantic_text, embedding)
  → save_resume_bundle (Postgres, same as /api/parse persistence)

Requires ``DATABASE_URL`` in ``.env`` (and network access to the DB).

Examples:
  # Verify: insert first 3 rows only
  python -m data_sandbox.persist_jsonl_to_db --input data_sandbox/output.jsonl --limit 3

  # Full file
  python -m data_sandbox.persist_jsonl_to_db --input data_sandbox/output.jsonl --limit 0

  # Validate + build bundles without writing
  python -m data_sandbox.persist_jsonl_to_db --input data_sandbox/output.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import settings
from pydantic import ValidationError
from schemas.models import ResumeStructured
from services.resume_storage_bundle import build_resume_storage_bundle
from storage.postgres_store import save_resume_bundle
from utils.errors import DatabaseError


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persist sandbox JSONL to Postgres via production bundle pipeline")
    p.add_argument(
        "--input",
        type=str,
        default="data_sandbox/output.jsonl",
        help="Path to JSONL (one ResumeStructured JSON object per line)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max rows to persist (default 3 for smoke test). Use 0 for all lines.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build bundles only; do not connect or write to the database",
    )
    p.add_argument(
        "--id-prefix",
        type=str,
        default="sandbox",
        help="resume_id format: {prefix}_{uuid32} (default sandbox)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.input)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and not settings.DATABASE_URL:
        print(
            "DATABASE_URL is not set. Add it to .env or export it before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    inserted: list[str] = []
    errors: list[tuple[int, str]] = []
    line_no = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line_no += 1
        line = raw_line.strip()
        if not line:
            continue
        if args.limit > 0 and len(inserted) >= args.limit:
            break

        try:
            data = json.loads(line)
            resume = ResumeStructured.model_validate(data)
            bundle = build_resume_storage_bundle(resume)
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append((line_no, str(exc)))
            continue
        except Exception as exc:
            errors.append((line_no, f"{type(exc).__name__}: {exc}"))
            continue

        resume_id = f"{args.id_prefix}_{uuid4().hex}"
        if args.dry_run:
            inserted.append(resume_id)
            print(f"  [dry-run] line {line_no} → resume_id={resume_id} OK")
            continue

        try:
            save_resume_bundle(
                resume_id=resume_id,
                bundle=bundle,
                source_file_name=path.name,
                source_file_type=".jsonl",
            )
        except DatabaseError as exc:
            errors.append((line_no, str(exc)))
            continue

        inserted.append(resume_id)
        sem_len = len(bundle.get("semantic_text") or "")
        emb = bundle.get("embedding")
        emb_note = f"{len(emb)}d" if emb else "null"
        print(f"  line {line_no} → {resume_id}  semantic_text={sem_len} chars  embedding={emb_note}")

    print()
    print(f"Done. Inserted: {len(inserted)}, errors: {len(errors)}")
    if errors:
        for ln, msg in errors:
            print(f"  line {ln}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

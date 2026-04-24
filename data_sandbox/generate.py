"""Main entry point for synthetic resume generation.

Usage:
    python -m data_sandbox.generate --count 9 --output data_sandbox/output.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data_sandbox.config import SandboxConfig
from data_sandbox.enrich import enrich_and_validate
from data_sandbox.skeleton import TIERS, build_skeleton


def _distribute_tiers(count: int) -> list[tuple[str, int]]:
    """Split *count* evenly across tiers (1:1:1), remainder goes to later tiers."""
    per_tier = count // len(TIERS)
    remainder = count % len(TIERS)
    result: list[tuple[str, int]] = []
    for i, tier in enumerate(TIERS):
        n = per_tier + (1 if i >= len(TIERS) - remainder else 0)
        result.append((tier, n))
    return result


def generate_resumes(cfg: SandboxConfig, out_file=None) -> list[dict]:
    tier_plan = _distribute_tiers(cfg.count)
    results: list[dict] = []
    index = 1

    for tier, n in tier_plan:
        for _ in range(n):
            skeleton = build_skeleton(index, tier)
            try:
                validated = enrich_and_validate(
                    skeleton,
                    max_repair=cfg.max_repair_attempts,
                    provider=cfg.llm_provider,
                    model=cfg.llm_model,
                )
                record = validated.model_dump()
                results.append(record)
                if out_file is not None:
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                print(f"  [{index}/{cfg.count}] {tier:8s} OK")
            except Exception as exc:
                print(f"  [{index}/{cfg.count}] {tier:8s} FAILED: {exc}", file=sys.stderr)
            index += 1

    return results


def main() -> None:
    cfg = SandboxConfig.from_cli()
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating {cfg.count} synthetic resumes (1:1:1 tier split); "
        f"appending each OK row to {cfg.output} as it finishes.",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        records = generate_resumes(cfg, out_file=f)
    print(f"Done. {len(records)}/{cfg.count} resumes → {cfg.output}")


if __name__ == "__main__":
    main()

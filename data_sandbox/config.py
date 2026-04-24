from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxConfig:
    count: int = 9
    output: str = "data_sandbox/output.jsonl"
    max_repair_attempts: int = 2
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    @classmethod
    def from_cli(cls) -> SandboxConfig:
        parser = argparse.ArgumentParser(
            description="Generate synthetic resume raw_json data",
        )
        parser.add_argument(
            "--count", type=int, default=9,
            help="Total resumes to generate (split 1:1:1 across tiers)",
        )
        parser.add_argument(
            "--output", type=str, default="data_sandbox/output.jsonl",
            help="Output JSONL path",
        )
        parser.add_argument(
            "--max-repair", type=int, default=2,
            help="Max validation-repair attempts per resume",
        )
        parser.add_argument("--provider", type=str, default=None)
        parser.add_argument("--model", type=str, default=None)
        args = parser.parse_args()
        return cls(
            count=args.count,
            output=args.output,
            max_repair_attempts=args.max_repair,
            llm_provider=args.provider,
            llm_model=args.model,
        )

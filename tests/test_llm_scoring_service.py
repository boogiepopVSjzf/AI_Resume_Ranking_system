#python -m tests.test_llm_scoring_service
from __future__ import annotations

import json

from services.llm_scoring_service import LLMScoringService


def main() -> None:
    """
    Minimal local test for LLMScoringService.

    Run with:
        python -m tests.test_llm_scoring_service
    """

    resumes = [
        {
            "candidate_id": "cand_001",
            "raw_pdf_path": "storage/resumes/cand_001.pdf",
            "resume_text": (
                "Claire Fu has experience in Python, SQL, machine learning, "
                "statistics, NLP, research, and data analysis. "
                "She has completed multiple projects involving predictive modeling "
                "and communication of results."
            ),
        },
        {
            "candidate_id": "cand_002",
            "raw_pdf_path": "storage/resumes/cand_002.pdf",
            "resume_text": (
                "Candidate has experience with Excel, reporting, and general office support. "
                "Limited evidence of Python or SQL. Some communication skills are present."
            ),
        },
    ]

    jd = {
        "job_id": "jd_001",
        "jd_text": (
            "We are hiring a Data Scientist. "
            "Required: Python and SQL. "
            "Preferred: machine learning, statistics, NLP, experimentation, "
            "and communication with stakeholders."
        ),
    }

    service = LLMScoringService()

    results = service.score_resumes_against_jd(
        resumes=resumes,
        jd=jd,
        schema_key="data_scientist_v1",
        provider="gemini",
        model="gemini-2.5-flash",
    )

    print(json.dumps([item.model_dump() for item in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
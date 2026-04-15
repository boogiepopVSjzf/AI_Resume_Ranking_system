from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from schemas.final_result import CandidateFinalResult
from schemas.llm_explain import ExplanationOutput
from schemas.llm_score import ScoreOutput
from services.llm_service import LLMStructuredClient
from services.schema_registry_service import get_schema_bundle


class LLMScoringService:
    """
    Score one or more resume JSON objects against one JD JSON object.

    This service resolves score / explanation schemas internally from schema_key,
    then uses the structured LLM client to generate:
    1. ScoreOutput
    2. ExplanationOutput
    """

    def __init__(self, structured_client: Optional[LLMStructuredClient] = None) -> None:
        """
        Initialize the scoring service.

        If a structured_client is provided, it will be used as the default client.
        Provider/model overrides can still be applied at runtime by creating
        a temporary client with the requested provider/model.
        """
        self.structured_client = structured_client or LLMStructuredClient()

    def score_resumes_against_jd(
        self,
        resumes: List[Dict[str, Any]],
        jd: Dict[str, Any],
        schema_key: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CandidateFinalResult]:
        """
        Score multiple resumes against one JD.

        Args:
            resumes: Resume JSON objects.
            jd: JD JSON object.
            schema_key: Preset schema bundle key.
            provider: Optional runtime LLM provider override.
            model: Optional runtime LLM model override.

        Returns:
            A list of CandidateFinalResult objects.
        """
        if not resumes:
            raise ValueError("resumes must not be empty")

        bundle = get_schema_bundle(schema_key)

        job_id = self._extract_job_id(jd)
        jd_text = self._extract_jd_text(jd)

        results: List[CandidateFinalResult] = []

        for index, resume in enumerate(resumes):
            candidate_id = self._extract_candidate_id(resume, index=index)
            raw_pdf_path = self._extract_raw_pdf_path(resume)
            resume_text = self._extract_resume_text(resume)

            score_output = self._generate_score_output_llm(
                candidate_id=candidate_id,
                job_id=job_id,
                resume_json=resume,
                jd_json=jd,
                resume_text=resume_text,
                jd_text=jd_text,
                score_schema_json=bundle.score_schema.model_dump(),
                provider=provider,
                model=model,
            )

            explanation_output = self._generate_explanation_output_llm(
                candidate_id=candidate_id,
                job_id=job_id,
                resume_json=resume,
                jd_json=jd,
                score_output=score_output,
                explanation_schema_json=bundle.explanation_schema.model_dump(),
                provider=provider,
                model=model,
            )

            results.append(
                CandidateFinalResult(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    score=score_output,
                    raw_pdf_path=raw_pdf_path,
                    explanation=explanation_output,
                )
            )

        return results

    def score_one_resume_against_jd(
        self,
        resume: Dict[str, Any],
        jd: Dict[str, Any],
        schema_key: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> CandidateFinalResult:
        """
        Convenience wrapper for scoring a single resume.
        """
        return self.score_resumes_against_jd(
            resumes=[resume],
            jd=jd,
            schema_key=schema_key,
            provider=provider,
            model=model,
        )[0]

    def _generate_score_output_llm(
        self,
        candidate_id: str,
        job_id: str,
        resume_json: Dict[str, Any],
        jd_json: Dict[str, Any],
        resume_text: str,
        jd_text: str,
        score_schema_json: Dict[str, Any],
        provider: Optional[str],
        model: Optional[str],
    ) -> ScoreOutput:
        """
        Use LLM to generate ScoreOutput.
        """
        client = self._get_runtime_client(provider=provider, model=model)

        task_instruction = self._build_score_task_instruction()

        input_sections = {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "resume_json": resume_json,
            "resume_text": resume_text,
            "jd_json": jd_json,
            "jd_text": jd_text,
            "score_schema_definition": score_schema_json,
        }

        validated, usage = client.generate_structured(
            task_instruction=task_instruction,
            response_model=ScoreOutput,
            input_sections=input_sections,
        )

        # Force critical identity fields to remain aligned with the request context.
        validated.candidate_id = candidate_id
        validated.job_id = job_id
        validated.llm_usage = usage or {}

        return validated

    def _generate_explanation_output_llm(
        self,
        candidate_id: str,
        job_id: str,
        resume_json: Dict[str, Any],
        jd_json: Dict[str, Any],
        score_output: ScoreOutput,
        explanation_schema_json: Dict[str, Any],
        provider: Optional[str],
        model: Optional[str],
    ) -> ExplanationOutput:
        """
        Use LLM to generate ExplanationOutput.
        """
        client = self._get_runtime_client(provider=provider, model=model)

        task_instruction = self._build_explanation_task_instruction()

        input_sections = {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "resume_json": resume_json,
            "jd_json": jd_json,
            "score_output": score_output.model_dump(),
            "explanation_schema_definition": explanation_schema_json,
        }

        validated, _usage = client.generate_structured(
            task_instruction=task_instruction,
            response_model=ExplanationOutput,
            input_sections=input_sections,
        )

        # Force critical identity fields to remain aligned with the request context.
        validated.candidate_id = candidate_id
        validated.job_id = job_id

        return validated

    def _get_runtime_client(
        self,
        provider: Optional[str],
        model: Optional[str],
    ) -> LLMStructuredClient:
        """
        Build a runtime client.

        If provider/model are both omitted, reuse the default client.
        Otherwise create a temporary client with the requested runtime overrides.
        """
        if provider is None and model is None:
            return self.structured_client

        return LLMStructuredClient(
            provider=provider,
            model=model,
        )

    def _build_score_task_instruction(self) -> str:
        """
        Build task instruction for score generation.
        """
        return (
            "Evaluate exactly one candidate against exactly one job description. "
            "Return a ScoreOutput JSON object only. "
            "Use the provided score_schema_definition as the scoring contract. "
            "Follow the schema fields and enumerations exactly. "
            "Use only evidence supported by the resume and job description. "
            "Do not invent qualifications, experience, projects, skills, licenses, or outcomes. "
            "For hard constraints, determine whether the requirement is met based on available evidence. "
            "For soft rules, assign score impact conservatively and consistently. "
            "For red flags, trigger them only when there is explicit supporting evidence. "
            "Keep dimension scores consistent with final_score and recommendation decision. "
            "Return valid JSON only."
        )

    def _build_explanation_task_instruction(self) -> str:
        """
        Build task instruction for explanation generation.
        """
        return (
            "Generate exactly one ExplanationOutput JSON object only. "
            "Use the provided explanation_schema_definition as the explanation contract. "
            "Ground every statement in the resume, job description, and score_output. "
            "Do not invent evidence. "
            "Strengths should reflect supported candidate advantages. "
            "Gaps should reflect missing or weak alignment with the JD. "
            "Red flags should only be included when supported by evidence or by the score_output. "
            "The rule_based_analysis should be consistent with the score output. "
            "The explanation must remain consistent with the recommendation decision and final score. "
            "Return valid JSON only."
        )

    def _extract_candidate_id(self, resume: Dict[str, Any], index: int) -> str:
        """
        Extract candidate id from resume JSON.
        """
        candidate_id = (
            resume.get("candidate_id")
            or resume.get("id")
            or resume.get("resume_id")
            or resume.get("profile_id")
        )
        if candidate_id is None:
            return f"candidate_{index + 1}"
        return str(candidate_id)

    def _extract_job_id(self, jd: Dict[str, Any]) -> str:
        """
        Extract job id from JD JSON.
        """
        job_id = jd.get("job_id") or jd.get("id") or jd.get("jd_id")
        if job_id is None:
            return "job_1"
        return str(job_id)

    def _extract_raw_pdf_path(self, resume: Dict[str, Any]) -> str:
        """
        Extract raw PDF path if available.
        """
        raw_pdf_path = (
            resume.get("raw_pdf_path")
            or resume.get("pdf_path")
            or resume.get("source_pdf")
            or resume.get("resume_pdf_path")
        )
        if raw_pdf_path is None:
            return ""
        return str(raw_pdf_path)

    def _extract_resume_text(self, resume: Dict[str, Any]) -> str:
        """
        Extract plain resume text if available.
        Fall back to serialized resume JSON.
        """
        resume_text = (
            resume.get("resume_text")
            or resume.get("text")
            or resume.get("raw_text")
            or resume.get("parsed_text")
            or resume.get("content")
        )
        if resume_text is not None:
            return str(resume_text)

        return json.dumps(resume, ensure_ascii=False, indent=2)

    def _extract_jd_text(self, jd: Dict[str, Any]) -> str:
        """
        Extract plain JD text if available.
        Fall back to serialized JD JSON.
        """
        jd_text = (
            jd.get("jd_text")
            or jd.get("text")
            or jd.get("job_description")
            or jd.get("description")
            or jd.get("content")
        )
        if jd_text is not None:
            return str(jd_text)

        return json.dumps(jd, ensure_ascii=False, indent=2)
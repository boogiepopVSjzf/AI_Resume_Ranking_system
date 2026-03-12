from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


DecisionType = Literal["PASS", "SOFT_FAIL", "HARD_FAIL"]


@dataclass
class ValidityStats:
    char_count: int
    word_count: int
    line_count: int
    non_empty_line_count: int
    section_hits: List[str] = field(default_factory=list)
    date_matches: int = 0
    email_found: bool = False
    phone_found: bool = False
    linkedin_found: bool = False
    github_found: bool = False
    bullet_lines: int = 0
    repeated_line_ratio: float = 0.0
    possible_jd_signals: int = 0


@dataclass
class ValidityResult:
    decision: DecisionType
    confidence: float
    overall_score: float
    component_scores: Dict[str, float]
    reasons: List[str]
    warnings: List[str]
    stats: ValidityStats


class ResumeValidityChecker:
    """
    Check whether a clean TXT input is sufficiently resume-like
    before sending it to an LLM extraction or ranking pipeline.
    """

    SECTION_KEYWORDS = {
        "education": [
            "education", "academic background", "academic history"
        ],
        "experience": [
            "experience", "work experience", "professional experience",
            "employment", "work history"
        ],
        "skills": [
            "skills", "technical skills", "core competencies", "competencies"
        ],
        "projects": [
            "projects", "project experience", "selected projects"
        ],
        "summary": [
            "summary", "professional summary", "profile", "objective"
        ],
        "research": [
            "research", "research experience"
        ],
        "publications": [
            "publications", "papers"
        ],
        "leadership": [
            "leadership", "leadership experience", "activities", "extracurricular activities"
        ],
        "certifications": [
            "certifications", "licenses", "certificates"
        ],
        "awards": [
            "awards", "honors"
        ],
    }

    JD_KEYWORDS = [
        "responsibilities",
        "requirements",
        "qualifications",
        "preferred qualifications",
        "job description",
        "about the role",
        "what you'll do",
        "what we are looking for",
        "minimum qualifications",
    ]

    DEGREE_KEYWORDS = [
        "b.s", "bs", "b.a", "ba", "m.s", "ms", "m.a", "ma",
        "ph.d", "phd", "bachelor", "master", "university", "college"
    ]

    SKILL_KEYWORDS = [
        "python", "r", "sql", "java", "c++", "machine learning",
        "deep learning", "statistics", "data analysis", "tableau",
        "excel", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy"
    ]

    ROLE_KEYWORDS = [
        "intern", "research assistant", "data scientist", "analyst",
        "software engineer", "student", "teaching assistant", "graduate assistant"
    ]

    DATE_PATTERNS = [
        re.compile(r"\b(19|20)\d{2}\b"),
        re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(19|20)\d{2}\b", re.I),
        re.compile(r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}\b"),
        re.compile(r"\b(19|20)\d{2}\s*[-–]\s*(present|current|now)\b", re.I),
    ]

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_PATTERN = re.compile(
        r"(?:\+?1[\s\-\.]?)?(?:\(?\d{3}\)?[\s\-\.]?)\d{3}[\s\-\.]?\d{4}\b"
    )
    LINKEDIN_PATTERN = re.compile(r"linkedin\.com|linkedin", re.I)
    GITHUB_PATTERN = re.compile(r"github\.com|github", re.I)

    BULLET_PATTERN = re.compile(r"^\s*[-*•▪◦]\s+")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        hard_fail_char_threshold: int = 120,
        soft_min_char_threshold: int = 250,
        strong_char_threshold: int = 500,
    ) -> None:
        self.hard_fail_char_threshold = hard_fail_char_threshold
        self.soft_min_char_threshold = soft_min_char_threshold
        self.strong_char_threshold = strong_char_threshold

    def check_file(self, txt_path: str | Path) -> ValidityResult:
        path = Path(txt_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.check_text(text)

    def check_text(self, text: str) -> ValidityResult:
        normalized_text = self._normalize_text(text)
        lines = [line.rstrip() for line in normalized_text.splitlines()]
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        stats = self._collect_stats(normalized_text, lines, non_empty_lines)

        component_scores: Dict[str, float] = {}
        reasons: List[str] = []
        warnings: List[str] = []

        length_score = self._score_length(stats, reasons, warnings)
        section_score = self._score_sections(stats, reasons, warnings)
        density_score = self._score_content_density(normalized_text, stats, reasons, warnings)
        contact_score = self._score_contact(stats, reasons, warnings)
        penalty_score = self._score_penalty(stats, reasons, warnings)

        component_scores["length_score"] = length_score
        component_scores["section_score"] = section_score
        component_scores["content_density_score"] = density_score
        component_scores["contact_score"] = contact_score
        component_scores["penalty_score"] = penalty_score

        overall_score = length_score + section_score + density_score + contact_score + penalty_score
        overall_score = max(0.0, min(100.0, overall_score))

        decision = self._make_decision(stats, overall_score)
        confidence = self._estimate_confidence(stats, overall_score, decision)

        return ValidityResult(
            decision=decision,
            confidence=confidence,
            overall_score=overall_score,
            component_scores=component_scores,
            reasons=self._deduplicate(reasons),
            warnings=self._deduplicate(warnings),
            stats=stats,
        )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize line endings and strip obvious control characters.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x00", " ")
        return text.strip()

    def _collect_stats(
        self,
        text: str,
        lines: List[str],
        non_empty_lines: List[str],
    ) -> ValidityStats:
        lowered_lines = [line.lower().strip() for line in non_empty_lines]
        lowered_text = text.lower()

        section_hits = self._detect_sections(non_empty_lines)
        date_matches = self._count_date_matches(text)
        email_found = bool(self.EMAIL_PATTERN.search(text))
        phone_found = bool(self.PHONE_PATTERN.search(text))
        linkedin_found = bool(self.LINKEDIN_PATTERN.search(text))
        github_found = bool(self.GITHUB_PATTERN.search(text))
        bullet_lines = sum(1 for line in non_empty_lines if self.BULLET_PATTERN.search(line))
        repeated_line_ratio = self._compute_repeated_line_ratio(lowered_lines)
        possible_jd_signals = sum(1 for kw in self.JD_KEYWORDS if kw in lowered_text)

        return ValidityStats(
            char_count=len(text),
            word_count=len(text.split()),
            line_count=len(lines),
            non_empty_line_count=len(non_empty_lines),
            section_hits=section_hits,
            date_matches=date_matches,
            email_found=email_found,
            phone_found=phone_found,
            linkedin_found=linkedin_found,
            github_found=github_found,
            bullet_lines=bullet_lines,
            repeated_line_ratio=repeated_line_ratio,
            possible_jd_signals=possible_jd_signals,
        )

    def _detect_sections(self, non_empty_lines: List[str]) -> List[str]:
        hits: List[str] = []
        for canonical_name, aliases in self.SECTION_KEYWORDS.items():
            if self._has_section_header(non_empty_lines, aliases):
                hits.append(canonical_name)
        return hits

    def _has_section_header(self, non_empty_lines: List[str], aliases: List[str]) -> bool:
        for i, line in enumerate(non_empty_lines):
            normalized_line = self.WHITESPACE_PATTERN.sub(" ", line.lower().strip())
            if len(normalized_line) > 50:
                continue
            for alias in aliases:
                if normalized_line == alias or normalized_line.startswith(alias + ":"):
                    if i + 1 < len(non_empty_lines):
                        return True
        return False

    def _count_date_matches(self, text: str) -> int:
        count = 0
        for pattern in self.DATE_PATTERNS:
            count += len(pattern.findall(text))
        return count

    def _compute_repeated_line_ratio(self, lowered_lines: List[str]) -> float:
        if not lowered_lines:
            return 0.0
        counts: Dict[str, int] = {}
        for line in lowered_lines:
            counts[line] = counts.get(line, 0) + 1

        repeated_instances = sum(count for count in counts.values() if count > 1)
        return repeated_instances / len(lowered_lines)

    def _score_length(
        self,
        stats: ValidityStats,
        reasons: List[str],
        warnings: List[str],
    ) -> float:
        if stats.char_count < self.hard_fail_char_threshold:
            warnings.append("Text is extremely short.")
            return 0.0
        if stats.char_count < self.soft_min_char_threshold:
            warnings.append("Text may be too short to support reliable extraction.")
            return 6.0
        if stats.char_count < self.strong_char_threshold:
            reasons.append("Text length is minimally sufficient.")
            return 13.0
        reasons.append("Sufficient content length detected.")
        return 20.0

    def _score_sections(
        self,
        stats: ValidityStats,
        reasons: List[str],
        warnings: List[str],
    ) -> float:
        n = len(stats.section_hits)
        if n == 0:
            warnings.append("No clear resume section headers detected.")
            return 0.0
        if n == 1:
            warnings.append("Only one resume-like section detected.")
            return 8.0
        if n == 2:
            reasons.append("Multiple resume-like sections detected.")
            return 16.0
        if n == 3:
            reasons.append("Good section coverage detected.")
            return 24.0
        reasons.append("Strong resume structure detected.")
        return 30.0

    def _score_content_density(
        self,
        text: str,
        stats: ValidityStats,
        reasons: List[str],
        warnings: List[str],
    ) -> float:
        lowered_text = text.lower()

        degree_hits = sum(1 for kw in self.DEGREE_KEYWORDS if kw in lowered_text)
        skill_hits = sum(1 for kw in self.SKILL_KEYWORDS if kw in lowered_text)
        role_hits = sum(1 for kw in self.ROLE_KEYWORDS if kw in lowered_text)

        score = 0.0

        if stats.date_matches >= 4:
            score += 8.0
        elif stats.date_matches >= 2:
            score += 5.0
        elif stats.date_matches >= 1:
            score += 2.0
        else:
            warnings.append("Few or no date patterns detected.")

        if stats.bullet_lines >= 6:
            score += 7.0
        elif stats.bullet_lines >= 3:
            score += 5.0
        elif stats.bullet_lines >= 1:
            score += 2.0
        else:
            warnings.append("Few or no bullet-like experience lines detected.")

        semantic_hits = 0
        if degree_hits > 0:
            semantic_hits += 1
        if skill_hits > 0:
            semantic_hits += 1
        if role_hits > 0:
            semantic_hits += 1

        if semantic_hits == 3:
            score += 10.0
            reasons.append("Education, skill, and role signals detected.")
        elif semantic_hits == 2:
            score += 7.0
            reasons.append("Moderate resume content density detected.")
        elif semantic_hits == 1:
            score += 4.0
        else:
            warnings.append("Weak resume-specific content density.")

        return min(score, 25.0)

    def _score_contact(
        self,
        stats: ValidityStats,
        reasons: List[str],
        warnings: List[str],
    ) -> float:
        score = 0.0

        if stats.email_found:
            score += 5.0
        if stats.phone_found:
            score += 2.0
        if stats.linkedin_found:
            score += 1.5
        if stats.github_found:
            score += 1.5

        if score == 0.0:
            warnings.append("No contact signal detected.")
        else:
            reasons.append("Contact information is present.")

        return min(score, 10.0)

    def _score_penalty(
        self,
        stats: ValidityStats,
        reasons: List[str],
        warnings: List[str],
    ) -> float:
        penalty = 0.0

        if stats.repeated_line_ratio > 0.35:
            penalty -= 8.0
            warnings.append("High repeated-line ratio detected.")
        elif stats.repeated_line_ratio > 0.2:
            penalty -= 4.0
            warnings.append("Moderate repeated-line ratio detected.")

        if stats.possible_jd_signals >= 3:
            penalty -= 7.0
            warnings.append("Text may resemble a job description more than a resume.")
        elif stats.possible_jd_signals >= 1:
            penalty -= 3.0

        return max(-15.0, penalty)

    def _make_decision(self, stats: ValidityStats, overall_score: float) -> DecisionType:
        """
        Apply hard rules first, then score-based decision.
        """
        if stats.char_count < self.hard_fail_char_threshold:
            return "HARD_FAIL"

        minimal_resume_signals = (
            len(stats.section_hits) > 0
            or stats.date_matches > 0
            or stats.email_found
            or stats.phone_found
            or stats.bullet_lines > 0
        )

        if not minimal_resume_signals:
            return "HARD_FAIL"

        if overall_score >= 65:
            return "PASS"
        if overall_score >= 40:
            return "SOFT_FAIL"
        return "HARD_FAIL"

    def _estimate_confidence(
        self,
        stats: ValidityStats,
        overall_score: float,
        decision: DecisionType,
    ) -> float:
        """
        Estimate a simple confidence score in [0, 1].
        """
        confidence = 0.5

        if len(stats.section_hits) >= 3:
            confidence += 0.15
        if stats.date_matches >= 3:
            confidence += 0.1
        if stats.email_found:
            confidence += 0.05
        if stats.bullet_lines >= 3:
            confidence += 0.1
        if stats.repeated_line_ratio > 0.3:
            confidence -= 0.1
        if stats.possible_jd_signals >= 3:
            confidence -= 0.15

        if decision == "PASS" and overall_score >= 75:
            confidence += 0.05
        if decision == "HARD_FAIL" and overall_score <= 20:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _deduplicate(self, items: List[str]) -> List[str]:
        seen = set()
        output = []
        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)
        return output
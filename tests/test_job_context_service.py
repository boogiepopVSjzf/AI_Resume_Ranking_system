from __future__ import annotations

import pytest
from unittest.mock import patch

from services.job_context_service import (
    JDSourceConflict,
    JobContextEmpty,
    build_job_context,
    resolve_jd_body,
)


class TestResolveJdBody:
    def test_text_only(self):
        assert resolve_jd_body("  Senior Python dev  ", None) == "Senior Python dev"

    def test_neither_returns_empty(self):
        assert resolve_jd_body("", None) == ""
        assert resolve_jd_body(None, None) == ""

    @patch("services.job_context_service.jd_text_from_pdf", return_value="Extracted JD")
    def test_pdf_only(self, mock_pdf):
        result = resolve_jd_body(None, b"%PDF-fake")
        assert result == "Extracted JD"
        mock_pdf.assert_called_once_with(b"%PDF-fake")

    @patch("services.job_context_service.jd_text_from_pdf", return_value="Extracted JD")
    def test_pdf_with_blank_text_is_fine(self, mock_pdf):
        result = resolve_jd_body("   ", b"%PDF-fake")
        assert result == "Extracted JD"

    def test_both_channels_raises(self):
        with pytest.raises(JDSourceConflict):
            resolve_jd_body("Some text", b"%PDF-fake")

    def test_whitespace_only_text_counts_as_empty(self):
        assert resolve_jd_body("   \n\t  ", None) == ""


class TestBuildJobContext:
    def test_both_present(self):
        ctx = build_job_context("请重点看项目经验", "We need a Python developer")
        assert ctx["hr_note"] == "请重点看项目经验"
        assert ctx["jd_text"] == "We need a Python developer"
        assert "HR_NOTE:" in ctx["merged_context"]
        assert "JOB_DESCRIPTION:" in ctx["merged_context"]
        assert "请重点看项目经验" in ctx["merged_context"]
        assert "We need a Python developer" in ctx["merged_context"]

    def test_only_hr_note(self):
        ctx = build_job_context("多关注学历", "")
        assert ctx["hr_note"] == "多关注学历"
        assert ctx["jd_text"] == ""
        assert "HR_NOTE:" in ctx["merged_context"]
        assert "JOB_DESCRIPTION:" not in ctx["merged_context"]

    def test_only_jd(self):
        ctx = build_job_context("", "Looking for ML engineer")
        assert ctx["hr_note"] == ""
        assert ctx["jd_text"] == "Looking for ML engineer"
        assert "JOB_DESCRIPTION:" in ctx["merged_context"]
        assert "HR_NOTE:" not in ctx["merged_context"]

    def test_both_empty_raises(self):
        with pytest.raises(JobContextEmpty):
            build_job_context("", "")

    def test_whitespace_only_raises(self):
        with pytest.raises(JobContextEmpty):
            build_job_context("   ", "  \n ")

    def test_sections_separated_by_blank_line(self):
        ctx = build_job_context("note", "jd")
        assert "\n\n" in ctx["merged_context"]

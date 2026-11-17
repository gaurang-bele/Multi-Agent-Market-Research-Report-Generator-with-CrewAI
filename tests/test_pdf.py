"""
tests/test_pdf.py

Automated tests for the ReportLab PDF generator in report/pdf_generator.py.
Verifies that:
  1. compile_report() successfully creates a valid PDF file.
  2. The output file is non-zero in size.
  3. The PDF binary starts with the correct PDF magic bytes (%PDF-).
  4. No exceptions are raised during compilation with standard input.
  5. The function handles edge cases gracefully (empty markdown, None stats_df).
  6. PDF compiles correctly when a Pandas stats DataFrame is provided.

Run with:
    pytest tests/test_pdf.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from report.pdf_generator import compile_report


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """
# Tech Sector Q3 Analysis

## Executive Summary

The global technology sector demonstrated strong resilience in Q3 2024, with leading companies
reporting **robust revenue growth** despite macroeconomic headwinds. Semiconductor demand surged
driven by AI infrastructure spending.

## Market Overview

The tech sector reached a combined market capitalization of **$12.5 trillion** in Q3 2024,
representing a *15% year-over-year increase*. Cloud computing, AI, and semiconductor segments
led growth.

## Key Players

| Company   | Revenue (B) | Market Cap (B) | YoY Growth |
|-----------|-------------|----------------|------------|
| Apple     | $394.3      | $2,800         | +8.1%      |
| Microsoft | $211.9      | $2,900         | +16.2%     |
| NVIDIA    | $60.9       | $1,100         | +122.4%    |
| Google    | $307.4      | $1,800         | +8.7%      |

## Financial Metrics & Statistical Analysis

Summary statistics show high variance in YoY growth rates, with NVIDIA as a significant outlier.
The **mean revenue** across the top 10 companies was $322.9B.

## Market Outlook & Strategic Recommendations

1. **AI Infrastructure**: Continued capital expenditure on GPU clusters will sustain NVIDIA's growth.
2. **Cloud Expansion**: Microsoft Azure and Google Cloud remain primary beneficiaries.
3. **Semiconductor Cycle**: Recovery signals suggest a positive outlook for Intel and Samsung.

## Methodology

Data sourced from public financial filings (10-Q, annual reports) and reputable financial databases.
Analysis conducted using Python Pandas statistical computing.
"""

@pytest.fixture
def sample_stats_df() -> pd.DataFrame:
    """Return a sample describe() DataFrame for the PDF appendix."""
    data = {
        "Revenue_M": [394330, 211915, 307394, 574785, 134902],
        "Profit_M":  [96995,  72361,  73795,  30425,  39098],
        "MarketCap_B": [2800, 2900, 1800, 1700, 1200],
    }
    df = pd.DataFrame(data)
    return df.describe().round(2)


@pytest.fixture
def pdf_output_path(tmp_path) -> str:
    """Return a temporary path for PDF output."""
    return str(tmp_path / "test_market_report.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestCompileReport:
    """Tests for the compile_report() PDF generator function."""

    def test_pdf_file_created(self, pdf_output_path):
        """compile_report() must create a file at the specified output path."""
        result_path = compile_report(
            markdown_text=SAMPLE_MARKDOWN,
            query="Tech Sector Q3 Analysis",
            stats_df=None,
            output_path=pdf_output_path,
        )
        assert os.path.exists(result_path), f"PDF file was not created at {result_path}"

    def test_pdf_nonzero_size(self, pdf_output_path):
        """The generated PDF must not be empty."""
        compile_report(
            markdown_text=SAMPLE_MARKDOWN,
            query="Tech Sector Q3 Analysis",
            stats_df=None,
            output_path=pdf_output_path,
        )
        size = os.path.getsize(pdf_output_path)
        assert size > 1024, f"PDF is suspiciously small: {size} bytes"

    def test_pdf_valid_magic_bytes(self, pdf_output_path):
        """The PDF must start with the standard PDF magic bytes (%PDF-)."""
        compile_report(
            markdown_text=SAMPLE_MARKDOWN,
            query="Tech Sector Q3 Analysis",
            stats_df=None,
            output_path=pdf_output_path,
        )
        with open(pdf_output_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-", f"File does not start with PDF magic bytes: {header}"

    def test_pdf_no_exception_raised(self, pdf_output_path):
        """compile_report() must complete without raising any exception."""
        try:
            compile_report(
                markdown_text=SAMPLE_MARKDOWN,
                query="Tech Sector Q3 Analysis",
                stats_df=None,
                output_path=pdf_output_path,
            )
        except Exception as exc:
            pytest.fail(f"compile_report() raised an unexpected exception: {exc}")

    def test_pdf_with_stats_df(self, pdf_output_path, sample_stats_df):
        """PDF compilation must succeed when a Pandas stats DataFrame is provided."""
        result_path = compile_report(
            markdown_text=SAMPLE_MARKDOWN,
            query="Tech Sector Q3 Analysis",
            stats_df=sample_stats_df,
            output_path=pdf_output_path,
        )
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 1024

    def test_pdf_with_empty_markdown(self, pdf_output_path):
        """PDF compilation must not raise on empty markdown input."""
        try:
            compile_report(
                markdown_text="",
                query="Empty Report Test",
                stats_df=None,
                output_path=pdf_output_path,
            )
        except Exception as exc:
            pytest.fail(f"compile_report() raised with empty markdown: {exc}")
        assert os.path.exists(pdf_output_path)

    def test_pdf_with_tables_in_markdown(self, pdf_output_path):
        """PDF must compile correctly when Markdown contains pipe tables."""
        md_with_table = (
            "# Report\n\n"
            "## Key Metrics\n\n"
            "| Company | Revenue |\n"
            "|---------|----------|\n"
            "| Apple   | $394B    |\n"
            "| Google  | $307B    |\n\n"
            "End of report.\n"
        )
        result_path = compile_report(
            markdown_text=md_with_table,
            query="Table Test",
            stats_df=None,
            output_path=pdf_output_path,
        )
        assert os.path.exists(result_path)
        with open(result_path, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_returns_absolute_path(self, pdf_output_path):
        """compile_report() must return the absolute path of the created file."""
        result = compile_report(
            markdown_text=SAMPLE_MARKDOWN,
            query="Path Test",
            stats_df=None,
            output_path=pdf_output_path,
        )
        assert os.path.isabs(result), f"Returned path is not absolute: {result}"

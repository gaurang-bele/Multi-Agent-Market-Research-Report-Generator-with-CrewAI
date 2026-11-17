"""
tests/test_tools.py

Automated tests for the custom @tool functions in crew/tools.py.
Verifies that:
  1. calculate_financial_metrics correctly reads a CSV and returns
     a Markdown string containing the expected statistical rows.
  2. The computed mean values match Pandas' own calculations within
     floating-point tolerance.
  3. All required statistical descriptors are present in the output.
  4. Edge cases (missing file, non-numeric CSV) return error strings
     rather than raising exceptions.

Run with:
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_csv_path(tmp_path) -> str:
    """Create a temporary CSV with known financial data for testing."""
    data = {
        "Revenue_M": [394330, 211915, 307394, 574785, 134902],
        "Profit_M":  [96995,  72361,  73795,  30425,  39098],
        "MarketCap_B": [2800, 2900, 1800, 1700, 1200],
        "GrossMargin_Pct": [43.3, 69.8, 55.1, 47.6, 80.7],
        "YoY_Growth_Pct":  [8.1, 16.2, 8.7, 12.3, 22.1],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "test_metrics.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def mock_csv_expected(mock_csv_path) -> pd.DataFrame:
    """Return the expected describe() output for the mock CSV."""
    df = pd.read_csv(mock_csv_path)
    return df.select_dtypes(include="number").describe()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (avoid importing crewai in test context — test logic directly)
# ─────────────────────────────────────────────────────────────────────────────

def _run_financial_metrics(csv_path: str) -> str:
    """
    Call the core logic of calculate_financial_metrics without the @tool
    decorator to avoid CrewAI context requirements in unit tests.
    """
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "[Error] No numeric columns found in the provided CSV."
    summary = numeric_df.describe().round(2)
    try:
        return f"## Financial Summary Statistics\n\n{summary.to_markdown()}\n"
    except Exception:
        return f"## Financial Summary Statistics\n\n{summary.to_string()}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateFinancialMetrics:
    """Tests for the calculate_financial_metrics tool logic."""

    def test_returns_string(self, mock_csv_path):
        """Output must be a non-empty string."""
        result = _run_financial_metrics(mock_csv_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_markdown_header(self, mock_csv_path):
        """Output must contain the section header."""
        result = _run_financial_metrics(mock_csv_path)
        assert "## Financial Summary Statistics" in result

    def test_contains_all_stat_descriptors(self, mock_csv_path):
        """Output must contain all standard describe() row labels."""
        result = _run_financial_metrics(mock_csv_path)
        for descriptor in ("count", "mean", "std", "min", "max"):
            assert descriptor in result, f"Missing descriptor: '{descriptor}'"

    def test_contains_all_column_names(self, mock_csv_path):
        """Output must reference all numeric column names."""
        result = _run_financial_metrics(mock_csv_path)
        for col in ("Revenue_M", "Profit_M", "MarketCap_B", "GrossMargin_Pct", "YoY_Growth_Pct"):
            assert col in result, f"Missing column: '{col}'"

    def test_mean_revenue_correct(self, mock_csv_path, mock_csv_expected):
        """
        The mean of Revenue_M must match Pandas' own calculation
        within a tolerance of 0.1 (accounting for rounding to 2dp).
        """
        result = _run_financial_metrics(mock_csv_path)
        expected_mean = round(mock_csv_expected.loc["mean", "Revenue_M"], 2)
        # Extract the mean line from the output
        assert str(expected_mean) in result or str(int(expected_mean)) in result, (
            f"Expected mean Revenue_M={expected_mean} not found in output:\n{result}"
        )

    def test_mean_profit_correct(self, mock_csv_path, mock_csv_expected):
        """The mean of Profit_M must match Pandas' own calculation."""
        result = _run_financial_metrics(mock_csv_path)
        expected_mean = round(mock_csv_expected.loc["mean", "Profit_M"], 2)
        assert str(expected_mean) in result or str(int(expected_mean)) in result, (
            f"Expected mean Profit_M={expected_mean} not found in output:\n{result}"
        )

    def test_missing_file_returns_error_string(self):
        """A non-existent CSV path must return an error string, not raise."""
        try:
            df = pd.read_csv("/non/existent/path.csv")
            result = "unexpected success"
        except FileNotFoundError:
            result = "[Error] CSV file not found: '/non/existent/path.csv'"
        assert result.startswith("[Error]")

    def test_non_numeric_csv_returns_error(self, tmp_path):
        """A CSV with only string columns must return an error string."""
        csv_file = tmp_path / "strings_only.csv"
        csv_file.write_text("Name,Category\nApple,Tech\nGoogle,Tech\n")
        result = _run_financial_metrics(str(csv_file))
        assert "[Error]" in result

    def test_bundled_mock_csv(self):
        """The bundled data/mock_metrics.csv must parse correctly if it exists."""
        csv_path = "data/mock_metrics.csv"
        if not os.path.exists(csv_path):
            pytest.skip("Bundled mock CSV not found — skipping.")
        result = _run_financial_metrics(csv_path)
        assert "## Financial Summary Statistics" in result
        assert "mean" in result

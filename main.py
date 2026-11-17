"""
main.py

CLI entrypoint for the Multi-Agent Market Research & Report Generator.

Usage:
    python main.py --query "Tech Sector Q3 Analysis"
    python main.py --query "AI Startups 2024 Funding" --output output/ai_report.pdf

The script will:
  1. Load environment variables from .env
  2. Kick off the hierarchical CrewAI pipeline
  3. Compile the final Markdown into a styled PDF via ReportLab
  4. Write per-task execution benchmarks to benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment before any crewai / openai imports
load_dotenv()

from crew import MarketResearchCrew
from report import compile_report


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="market_research",
        description=(
            "Multi-Agent Market Research & Report Generator — "
            "Powered by CrewAI + GPT-4o"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "Tech Sector Q3 Analysis"
  python main.py --query "EV Market 2024 Overview" --output output/ev_report.pdf
  python main.py --query "Global SaaS Trends" --csv data/mock_metrics.csv
        """,
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Tech Sector Q3 Analysis",
        help="The market research topic / query to investigate (default: 'Tech Sector Q3 Analysis')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/market_report.pdf",
        help="Path for the output PDF file (default: output/market_report.pdf)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a CSV file for financial metrics analysis",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="benchmarks.json",
        help="Path for the benchmarks JSON output (default: benchmarks.json)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Environment Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_env() -> None:
    """Fail early with a clear error if required API keys are missing."""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY") and not os.getenv("SERPER_API_KEY"):
        missing.append("TAVILY_API_KEY (or SERPER_API_KEY)")

    if missing:
        print("\n❌  Missing required environment variables:")
        for key in missing:
            print(f"    • {key}")
        print("\nCreate a .env file from .env.example and fill in your API keys.")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks Writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_benchmarks(
    benchmark_log: list[dict],
    query: str,
    output_path: str,
    csv_path: str | None,
    benchmarks_file: str,
) -> None:
    """Serialise execution benchmarks to a JSON file."""
    payload = {
        "query": query,
        "output_pdf": output_path,
        "csv_input": csv_path,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_routing_history": benchmark_log,
        "summary": {
            "total_tasks_logged": len(benchmark_log),
            "total_duration_seconds": round(
                sum(e["duration_seconds"] for e in benchmark_log), 3
            ),
        },
    }

    with open(benchmarks_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n📊 Benchmarks written → {os.path.abspath(benchmarks_file)}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional CSV → Pandas Stats
# ─────────────────────────────────────────────────────────────────────────────

def _load_stats_df(csv_path: str | None) -> pd.DataFrame | None:
    """Load a CSV and return its numeric describe() DataFrame, or None."""
    if csv_path is None:
        # Use the bundled mock CSV if it exists
        default_csv = Path("data/mock_metrics.csv")
        if default_csv.exists():
            csv_path = str(default_csv)
        else:
            return None

    try:
        df = pd.read_csv(csv_path)
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return None
        return numeric_df.describe().round(2)
    except Exception as exc:
        print(f"⚠️  Could not load CSV for stats table: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("  Multi-Agent Market Research & Report Generator")
    print("  CrewAI · GPT-4o Manager · Hierarchical Process")
    print("=" * 60)

    # Validate environment before spending time
    _validate_env()

    # ── Run CrewAI Pipeline ─────────────────────────────────────────────────
    print(f"\n🚀 Initialising CrewAI pipeline for query: '{args.query}'")
    crew = MarketResearchCrew(query=args.query)

    pipeline_start = time.perf_counter()
    result = crew.run()
    pipeline_elapsed = time.perf_counter() - pipeline_start

    markdown_output: str = result.raw if hasattr(result, "raw") else str(result)
    print(f"\n✅ CrewAI pipeline completed in {pipeline_elapsed:.1f}s")

    # ── Save raw Markdown ───────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    md_path = Path(args.output).with_suffix(".md")
    md_path.write_text(markdown_output, encoding="utf-8")
    print(f"📝 Markdown report saved → {md_path.resolve()}")

    # ── Load stats DataFrame for PDF appendix ──────────────────────────────
    stats_df = _load_stats_df(args.csv)

    # ── Compile PDF ─────────────────────────────────────────────────────────
    print(f"\n📄 Compiling PDF report → {args.output}")
    pdf_path = compile_report(
        markdown_text=markdown_output,
        query=args.query,
        stats_df=stats_df,
        output_path=args.output,
    )

    # ── Write Benchmarks ────────────────────────────────────────────────────
    _write_benchmarks(
        benchmark_log=crew.benchmark_log,
        query=args.query,
        output_path=pdf_path,
        csv_path=args.csv,
        benchmarks_file=args.benchmarks,
    )

    # ── Final Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Run Complete!")
    print(f"  PDF Report  : {pdf_path}")
    print(f"  Markdown    : {md_path.resolve()}")
    print(f"  Benchmarks  : {os.path.abspath(args.benchmarks)}")
    print(f"  Duration    : {pipeline_elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

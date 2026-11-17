"""
crew/tasks.py

Defines the three CrewAI tasks bound to their respective agents.
Each task has a clear description, expected_output, and feeds its
output as context into the next task in the pipeline.
"""

from __future__ import annotations

from crewai import Task


def build_research_task(researcher_agent, query: str) -> Task:
    """
    Task 1 — Market Data Research
    The researcher searches the web and scrapes financial metrics for the query.
    """
    return Task(
        description=(
            f"Conduct comprehensive market research on: '{query}'.\n\n"
            "Your objectives:\n"
            "1. Use the Web Scraper Tool to search for financial metrics, market size, "
            "   key players, revenue figures, growth rates, and recent news.\n"
            "2. Scrape at least 3-5 authoritative sources (news sites, financial portals, "
            "   company investor relations pages).\n"
            "3. Extract specific numeric data points: revenue, profit margins, market cap, "
            "   YoY growth rates, and any available quarterly metrics.\n"
            "4. Present findings as clearly structured Markdown with source URLs cited.\n\n"
            "Output must be a structured Markdown document with sections: "
            "Market Overview, Key Players, Numeric Metrics, Sources."
        ),
        expected_output=(
            "A structured Markdown research report containing:\n"
            "- ## Market Overview section with narrative summary\n"
            "- ## Key Players section listing major companies with descriptions\n"
            "- ## Numeric Metrics section with a Markdown table of financial data points "
            "  (company, revenue, market cap, profit margin, YoY growth)\n"
            "- ## Sources section with cited URLs\n"
            "Minimum 500 words of substantive research content."
        ),
        agent=researcher_agent,
        output_file="output/research_output.md",
    )


def build_analysis_task(analyst_agent, research_task: Task) -> Task:
    """
    Task 2 — Financial Data Analysis
    The analyst takes research output, structures metrics into a CSV-equivalent
    DataFrame, computes statistics, and returns a Markdown summary table.
    """
    return Task(
        description=(
            "Analyse the market research data provided by the Lead Researcher.\n\n"
            "Your objectives:\n"
            "1. Extract all numeric financial metrics from the research output "
            "   (revenue, market cap, profit margins, growth rates etc.).\n"
            "2. If a CSV data file exists at 'data/mock_metrics.csv', use the "
            "   Financial Metrics Calculator tool to compute descriptive statistics.\n"
            "3. Manually compute or verify: mean, min, max, standard deviation for each "
            "   key metric across the companies identified.\n"
            "4. Identify any outliers, inconsistencies, or data quality issues.\n"
            "5. Return a clean, well-formatted Markdown table of summary statistics "
            "   plus a brief written commentary on the findings.\n\n"
            "Be precise with numbers — round to 2 decimal places."
        ),
        expected_output=(
            "A Markdown document containing:\n"
            "- ## Statistical Summary section with a Markdown table showing "
            "  count/mean/std/min/max for each financial metric\n"
            "- ## Data Quality Notes section identifying any anomalies or caveats\n"
            "- ## Key Insights section with 3-5 bullet points of analytical conclusions\n"
            "All numbers rounded to 2 decimal places."
        ),
        agent=analyst_agent,
        context=[research_task],
        output_file="output/analysis_output.md",
    )


def build_writing_task(writer_agent, research_task: Task, analysis_task: Task) -> Task:
    """
    Task 3 — Report Writing
    The writer synthesises both upstream outputs into a professional final report.
    """
    return Task(
        description=(
            "Synthesise the market research findings and financial analysis into a "
            "complete, professional market research report.\n\n"
            "Your objectives:\n"
            "1. Write a compelling Executive Summary (200-300 words).\n"
            "2. Expand the Market Overview with strategic context and market dynamics.\n"
            "3. Present the Key Players section with competitive positioning analysis.\n"
            "4. Integrate the statistical summary tables from the analyst's output.\n"
            "5. Write a Competitive Landscape section comparing companies.\n"
            "6. Write an Outlook & Recommendations section with 3-5 actionable insights.\n"
            "7. Add a Methodology section explaining data sources and analysis approach.\n\n"
            "The report must be structured for a C-suite audience — professional, concise, "
            "and data-driven. Use Markdown headers (##, ###) for all sections."
        ),
        expected_output=(
            "A complete Markdown market research report (minimum 1000 words) with the sections:\n"
            "1. # [Report Title with Query and Date]\n"
            "2. ## Executive Summary\n"
            "3. ## Market Overview\n"
            "4. ## Key Players & Competitive Landscape\n"
            "5. ## Financial Metrics & Statistical Analysis (include the data tables)\n"
            "6. ## Market Outlook & Strategic Recommendations\n"
            "7. ## Methodology\n"
            "Professional tone, no placeholder text, all sections populated with real content."
        ),
        agent=writer_agent,
        context=[research_task, analysis_task],
        output_file="output/final_report.md",
    )

"""
crew/agents.py

Defines the three specialized CrewAI agents:
  - Lead Researcher Agent  (Market Research Specialist)
  - Data Analyst Agent     (Financial Data Analyst)
  - Content Writer Agent   (Technical Reports Writer)

Each agent is wired with appropriate tools and an LLM sourced from env vars.
"""

from __future__ import annotations

import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .tools import web_scraper_tool, calculate_financial_metrics

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM — used by individual agents (lighter model to save cost)
# ─────────────────────────────────────────────────────────────────────────────

def _agent_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Factories
# ─────────────────────────────────────────────────────────────────────────────

def build_researcher() -> Agent:
    """
    Lead Researcher Agent.
    Queries search networks, locates financial data, and scrapes company metrics.
    """
    return Agent(
        role="Market Research Specialist",
        goal=(
            "Query search networks, locate up-to-date financial and market data "
            "for the given sector or company, and scrape key company metrics such as "
            "revenue, market cap, profit margins, and growth rates from authoritative sources."
        ),
        backstory=(
            "You are a seasoned market intelligence analyst with over 10 years of experience "
            "gathering competitive intelligence across technology, finance, and consumer sectors. "
            "You have deep expertise in evaluating source credibility, cross-referencing data points "
            "from multiple outlets, and surfacing the most relevant metrics for financial analysis. "
            "You approach each research task with rigour and always cite your sources clearly."
        ),
        tools=[web_scraper_tool],
        llm=_agent_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        max_rpm=10,
    )


def build_analyst() -> Agent:
    """
    Data Analyst Agent.
    Parses data matrices using Pandas, computes averages, and verifies numeric consistency.
    """
    return Agent(
        role="Financial Data Analyst",
        goal=(
            "Parse the raw financial data matrices collected by the researcher, load them "
            "into structured Pandas DataFrames, calculate comprehensive statistical summaries "
            "(mean, min, max, standard deviation, percentiles), and verify numeric consistency "
            "and outliers before returning clean Markdown tables."
        ),
        backstory=(
            "You are a quantitative analyst who specialises in turning raw, unstructured financial "
            "data into clean, actionable metrics. With a background in data science and financial "
            "modelling, you are adept at identifying data quality issues, normalising figures across "
            "different units (millions vs billions), and producing publication-ready statistical "
            "summaries that drive investment decisions."
        ),
        tools=[calculate_financial_metrics],
        llm=_agent_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        max_rpm=10,
    )


def build_writer() -> Agent:
    """
    Content Writer Agent.
    Synthesises information, writes sections, and structures the final report copy.
    """
    return Agent(
        role="Technical Reports Writer",
        goal=(
            "Synthesise the market research findings and quantitative analysis into a professional, "
            "well-structured written report. Write clear executive summaries, detailed section narratives, "
            "and actionable conclusions suitable for a C-suite audience."
        ),
        backstory=(
            "You are a senior technical writer and financial journalist who has authored hundreds of "
            "market research reports for investment banks, consulting firms, and technology enterprises. "
            "You excel at translating complex financial data into compelling, clearly-structured prose "
            "that is both technically precise and accessible to business decision-makers. "
            "Your reports always follow a logical structure: executive summary → market overview → "
            "key metrics → competitive landscape → outlook and recommendations."
        ),
        tools=[],  # Writer synthesises from context — no external tool calls needed
        llm=_agent_llm(),
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        max_rpm=10,
    )

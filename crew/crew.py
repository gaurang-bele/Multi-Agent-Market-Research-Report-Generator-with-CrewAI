"""
crew/crew.py

Assembles the MarketResearchCrew:
  - Three specialized agents (Researcher, Analyst, Writer)
  - Three chained tasks
  - Hierarchical process with a GPT-4o manager LLM
  - Step callback for per-task benchmarking
"""

from __future__ import annotations

import os
import time
from typing import Any

from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .agents import build_researcher, build_analyst, build_writer
from .tasks import build_research_task, build_analysis_task, build_writing_task

load_dotenv()


class MarketResearchCrew:
    """
    Encapsulates the full CrewAI multi-agent pipeline.

    Usage:
        crew = MarketResearchCrew(query="Tech Sector Q3 Analysis")
        result = crew.run()
        # result.raw contains the final report Markdown
        # crew.benchmark_log contains per-task timing data
    """

    def __init__(self, query: str) -> None:
        self.query = query
        self.benchmark_log: list[dict[str, Any]] = []
        self._task_start_times: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Manager LLM — GPT-4o (or configurable via env)
    # ─────────────────────────────────────────────────────────────────────────

    def _manager_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=os.getenv("MANAGER_LLM_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,  # low temperature for deterministic task delegation
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step callback — fires after each agent step for benchmarking
    # ─────────────────────────────────────────────────────────────────────────

    def _step_callback(self, step_output: Any) -> None:
        """Called by CrewAI after each agent step to record timing."""
        agent_role = getattr(step_output, "agent", "unknown")
        # Use agent role as key; first time = start, second = end
        if agent_role not in self._task_start_times:
            self._task_start_times[agent_role] = time.perf_counter()
        else:
            duration = time.perf_counter() - self._task_start_times.pop(agent_role)
            self.benchmark_log.append(
                {
                    "agent": str(agent_role),
                    "duration_seconds": round(duration, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Crew Assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _build_crew(self) -> Crew:
        # Instantiate agents
        researcher = build_researcher()
        analyst = build_analyst()
        writer = build_writer()

        # Build tasks with context chaining
        research_task = build_research_task(researcher, self.query)
        analysis_task = build_analysis_task(analyst, research_task)
        writing_task = build_writing_task(writer, research_task, analysis_task)

        return Crew(
            agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.hierarchical,
            manager_llm=self._manager_llm(),
            verbose=True,
            step_callback=self._step_callback,
            memory=False,           # keep deterministic for benchmarking
            max_rpm=20,             # global rate limit
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public run method
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Any:
        """
        Kick off the hierarchical crew pipeline.

        Returns:
            CrewOutput object. Use `.raw` for the final Markdown string.
        """
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

        crew = self._build_crew()

        print(f"\n{'='*60}")
        print(f"  Market Research Crew — Initiating Pipeline")
        print(f"  Query : {self.query}")
        print(f"  Manager LLM : {os.getenv('MANAGER_LLM_MODEL', 'gpt-4o')}")
        print(f"  Process : Hierarchical")
        print(f"{'='*60}\n")

        pipeline_start = time.perf_counter()
        result = crew.kickoff(inputs={"query": self.query})
        pipeline_duration = time.perf_counter() - pipeline_start

        # Append overall pipeline benchmark
        self.benchmark_log.append(
            {
                "agent": "PIPELINE_TOTAL",
                "duration_seconds": round(pipeline_duration, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        return result

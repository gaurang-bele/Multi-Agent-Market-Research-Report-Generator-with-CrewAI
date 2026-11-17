# Multi-Agent Market Research & Report Generator

> **CrewAI · GPT-4o · Pandas · ReportLab · Tavily**  
> Intermediate Assessment Project — Practical Multi-Agent Build

A production-quality Python application that orchestrates **three specialized AI agents** (Lead Researcher, Data Analyst, Content Writer) under a **hierarchical CrewAI manager (GPT-4o)** to automatically:
- 🔍 **Search & scrape** real-time market data via Tavily
- 📊 **Analyse metrics** with Pandas statistical summaries
- 📝 **Generate a structured report** in Markdown
- 📄 **Compile a styled PDF** with ReportLab (header banner, tables, page numbers)
- ⏱️ **Log execution benchmarks** to `benchmarks.json`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-4o Manager LLM                       │
│              (Process.hierarchical orchestrator)            │
└───────────┬─────────────────┬───────────────┬──────────────┘
            │                 │               │
    ┌───────▼──────┐  ┌───────▼──────┐  ┌────▼──────────┐
    │ Lead          │  │ Data         │  │ Content       │
    │ Researcher    │  │ Analyst      │  │ Writer        │
    │               │  │              │  │               │
    │ web_scraper   │  │ calculate_   │  │ (synthesises  │
    │ _tool         │  │ financial_   │  │ from context) │
    │ [Tavily API]  │  │ metrics      │  │               │
    └───────────────┘  │ [Pandas]     │  └───────────────┘
                       └──────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  ReportLab PDF     │
                    │  Compiler          │
                    │  output/market_    │
                    │  report.pdf        │
                    └────────────────────┘
```

---

## Project Structure

```
tayana_project/
├── crew/
│   ├── __init__.py        # Package exports
│   ├── agents.py          # Three agent factory functions
│   ├── tasks.py           # Three chained task definitions
│   ├── tools.py           # @tool: web_scraper_tool, calculate_financial_metrics
│   └── crew.py            # MarketResearchCrew with hierarchical Process
├── report/
│   ├── __init__.py
│   └── pdf_generator.py   # ReportLab A4 PDF compiler
├── tests/
│   ├── test_tools.py      # pytest: Pandas metrics validation
│   └── test_pdf.py        # pytest: PDF compilation integrity
├── sample_outputs/
│   ├── sample_report.md   # Pre-generated Markdown report
│   └── market_report.pdf  # Pre-compiled PDF (no API keys needed)
├── data/
│   └── mock_metrics.csv   # Mock dataset (10 tech companies)
├── output/                # Runtime outputs (gitignored)
├── main.py                # CLI entrypoint
├── requirements.txt
├── .env.example           # API key template
└── README.md
```

---

## Prerequisites

- Python 3.10+
- OpenAI API key (GPT-4o access)
- Tavily API key (free tier available at [tavily.com](https://tavily.com))

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/gaurang-bele/Multi-Agent-Market-Research-Report-Generator-with-CrewAI.git
cd Multi-Agent-Market-Research-Report-Generator-with-CrewAI
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example env file and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here
MANAGER_LLM_MODEL=gpt-4o
AGENT_LLM_MODEL=gpt-4o-mini
SEARCH_PROVIDER=tavily
```

---

## Running the Application

### Basic run (default query)
```bash
python main.py
```

### Custom research query
```bash
python main.py --query "Tech Sector Q3 Analysis"
```

### Full options
```bash
python main.py \
  --query "Global EV Market 2024" \
  --output output/ev_report.pdf \
  --csv data/mock_metrics.csv \
  --benchmarks benchmarks.json
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | `Tech Sector Q3 Analysis` | Research topic to investigate |
| `--output` | `output/market_report.pdf` | PDF output path |
| `--csv` | `data/mock_metrics.csv` | CSV for Pandas stats appendix |
| `--benchmarks` | `benchmarks.json` | Benchmark JSON output path |

### Expected output
```
============================================================
  Multi-Agent Market Research & Report Generator
  CrewAI · GPT-4o Manager · Hierarchical Process
============================================================

🚀 Initialising CrewAI pipeline for query: 'Tech Sector Q3 Analysis'
============================================================
  Market Research Crew — Initiating Pipeline
  Query : Tech Sector Q3 Analysis
  Manager LLM : gpt-4o
  Process : Hierarchical
============================================================

[Manager LLM] Delegating research task to Market Research Specialist...
[Researcher] Using Web Scraper Tool: Tech Sector Q3 Analysis...
[Manager LLM] Delegating analysis task to Financial Data Analyst...
[Analyst] Using Financial Metrics Calculator...
[Manager LLM] Delegating writing task to Technical Reports Writer...

✅ CrewAI pipeline completed in 94.3s
📝 Markdown report saved → output/market_report.md
✅ PDF report compiled successfully → output/market_report.pdf
📊 Benchmarks written → benchmarks.json

============================================================
  ✅  Run Complete!
  PDF Report  : output/market_report.pdf
  Markdown    : output/market_report.md
  Benchmarks  : benchmarks.json
  Duration    : 94.3s
============================================================
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_tools.py::TestCalculateFinancialMetrics::test_returns_string          PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_contains_markdown_header PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_contains_all_stat_descriptors PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_contains_all_column_names PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_mean_revenue_correct    PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_mean_profit_correct     PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_missing_file_returns_error_string PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_non_numeric_csv_returns_error PASSED
tests/test_tools.py::TestCalculateFinancialMetrics::test_bundled_mock_csv        PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_file_created                      PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_nonzero_size                      PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_valid_magic_bytes                 PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_no_exception_raised               PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_with_stats_df                     PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_with_empty_markdown               PASSED
tests/test_pdf.py::TestCompileReport::test_pdf_with_tables_in_markdown           PASSED
tests/test_pdf.py::TestCompileReport::test_returns_absolute_path                 PASSED

17 passed in X.XXs
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/market_report.pdf` | Compiled A4 PDF with header banner, tables, page numbers |
| `output/market_report.md` | Raw Markdown from Content Writer agent |
| `output/research_output.md` | Researcher agent output |
| `output/analysis_output.md` | Analyst agent output |
| `benchmarks.json` | Per-task timing + routing history |

---

## benchmarks.json Format

```json
{
  "query": "Tech Sector Q3 Analysis",
  "output_pdf": "output/market_report.pdf",
  "run_timestamp": "2024-11-18T00:00:00Z",
  "task_routing_history": [
    {"agent": "Market Research Specialist", "duration_seconds": 42.1, "timestamp": "..."},
    {"agent": "Financial Data Analyst",     "duration_seconds": 18.3, "timestamp": "..."},
    {"agent": "Technical Reports Writer",   "duration_seconds": 29.7, "timestamp": "..."},
    {"agent": "PIPELINE_TOTAL",             "duration_seconds": 90.1, "timestamp": "..."}
  ],
  "summary": {
    "total_tasks_logged": 4,
    "total_duration_seconds": 90.1
  }
}
```

---

## Sample Outputs

Pre-generated outputs (no API keys required) are in `sample_outputs/`:
- [`sample_report.md`](sample_outputs/sample_report.md) — Full Markdown report
- [`market_report.pdf`](sample_outputs/market_report.pdf) — Compiled PDF

---

## Git Commit History

| # | Commit | Description |
|---|--------|-------------|
| 1 | `chore: scaffold` | Project structure, `.gitignore`, `.env.example` |
| 2 | `feat: requirements` | All pinned dependencies |
| 3 | `feat: custom tools` | `web_scraper_tool` + `calculate_financial_metrics` |
| 4 | `feat: agents` | Three CrewAI agents with backstories and tool bindings |
| 5 | `feat: tasks + crew` | Chained tasks + hierarchical Crew assembly |
| 6 | `feat: PDF generator` | ReportLab with header, tables, page numbers |
| 7 | `feat: CLI entrypoint` | `main.py` with argparse + benchmarking |
| 8 | `test: pytest suite` | Tools + PDF compilation tests |
| 9 | `docs: README + CSV` | Setup instructions + mock dataset |
| 10 | `sample: outputs` | Pre-generated PDF + Markdown report |

---

## Troubleshooting

**`OPENAI_API_KEY` not set error**  
→ Ensure your `.env` file exists and contains a valid key.

**`TAVILY_API_KEY` not set error**  
→ Register at [tavily.com](https://tavily.com) for a free API key.

**PDF not generated**  
→ Check that the `output/` directory is writable. Run `pytest tests/test_pdf.py -v` to isolate PDF issues.

**Rate limit errors**  
→ The crew uses `max_rpm=20`. If you hit rate limits, increase `time.sleep` between retries or use a lower-tier model for agents (`AGENT_LLM_MODEL=gpt-4o-mini`).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

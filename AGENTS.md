# AGENTS.md

## Project Overview

**litradar** — Automated literature survey agent for academic research. Fetches papers from arXiv + Semantic Scholar, analyzes with LLM, generates structured survey reports.

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
# Edit config.yaml: add your DeepSeek API key
```

## Run

```bash
python run_survey.py
```

Outputs to `output/`:
- `survey_report.md` — structured literature review
- `papers.csv` — all papers with LLM-extracted metadata
- `citation_graph.json` — citation network
- `citation_graph.html` — interactive visualization (run `python gen_html.py` to regenerate)

## Architecture

```
run_survey.py          # Entry point: 4-stage pipeline
├── fetcher/           # Stage 1: Paper fetching
│   ├── arxiv_fetcher.py    # arXiv API (uses `arxiv` library)
│   └── s2_fetcher.py       # Semantic Scholar API (requests)
├── graph/             # Stage 2: Citation graph
│   └── citation_graph.py   # NetworkX graph + PageRank
├── analyzer/          # Stage 3: LLM analysis
│   ├── llm_client.py       # DeepSeek via OpenAI-compatible API
│   └── paper_analyzer.py   # Structured extraction + survey generation
└── reporter/          # Stage 4: Report generation
    ├── csv_exporter.py
    └── survey_writer.py
```

## Key Gotchas

### API Rate Limits
- **arXiv**: 429 errors common. Client sets `NO_PROXY` env vars and retries with backoff.
- **Semantic Scholar**: 100 req/5min without API key. Add `api_key` in config for higher limits.
- Both fetchers have built-in delays (`delay_seconds` in config).

### LLM Prompt Format
- `paper_analyzer.py` uses Python `.format()` with JSON examples in prompts.
- **Critical**: JSON examples must use double braces `{{` `}}` to escape literal braces, otherwise `.format()` throws `KeyError`.

### Proxy Issues (Windows)
- arXiv/S2 requests may fail behind corporate proxy.
- Fetchers set `NO_PROXY` env vars for known API hosts.
- If still failing, set system-level `NO_PROXY` or run with: `set NO_PROXY=export.arxiv.org,api.semanticscholar.org,api.deepseek.com`

### Output Size
- Analyzing 400+ papers with LLM is slow (~6s/paper). Pipeline caps at 80 papers by recency.
- Adjust `max_analyze` in `run_survey.py:74` or add filtering logic.

### HTML Visualization
- `gen_html.py` embeds JSON data directly into HTML (no server needed).
- Uses vis-network from CDN (`unpkg.com`). Requires internet to load.
- Regenerate after each survey run: `python gen_html.py`

## Config

Edit `config.yaml` (not tracked in git):
- `deepseek.api_key` — required
- `queries` — search terms (add/remove as needed)
- `arxiv.max_results` — per-query limit (default 200)
- `semantic_scholar.api_key` — optional, raises rate limit

## Testing

No test suite. Verify by running full pipeline and checking `output/` files.

## Conventions

- All modules use relative imports within packages (`from .module import Class`).
- Config is YAML, loaded once in `run_survey.py` and passed to components.
- Paper data flows as `List[Dict[str, Any]]` through the pipeline.
- LLM responses are parsed as JSON; failures fall back to default analysis dict.

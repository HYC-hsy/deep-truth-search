# Deep Truth Search

A research agent that finds supporting evidence for your claims — and learns where to look.

> [中文版](README_CN.md)

![Home](docs/screenshots/home.png)

## What it does

You have a claim. You need evidence.

Give it a claim, and it will:
1. Break it into multiple research angles (technical, academic, policy, etc.)
2. Search the web in parallel, visiting and reading actual pages
3. Score each source on five quality dimensions, dropping anything below threshold
4. Return structured arguments + evidence, each with source links and score breakdowns

You do the last part: read the evidence, make up your own mind.

---

## Design Philosophy

> **No judgments. Just evidence — lots of it, high quality.**

This isn't a tool that hands you conclusions. It searches deep, searches wide, and gives you traceable, scored evidence. You decide what it means.

**It works for your claim, not against it.**

If you type "X is the best mayor this city has ever had", the system:
- Won't tone it down to "X is a good mayor"
- Won't surface counterarguments because critics happen to be louder online
- Won't rewrite your claim into something more "balanced"

Whatever you put in, it faithfully finds supporting evidence. No rewording, no redirecting, no second-guessing.

---

## Key Features

### 1. Purpose-built for evidence gathering

Not general search. Not a chatbot. One job: you have a claim, it finds evidence.

It decomposes your claim into sub-angles and searches them in parallel — much broader coverage than a single query.

![Parallel search](docs/screenshots/searching-progress.png)

### 2. Five-dimension quality scoring

Found doesn't mean good. Each source gets scored by an LLM reviewer across five dimensions:

| Dimension | Max | What it measures |
|-----------|-----|------------------|
| **Authority** | 30 | Author credentials, institutional backing |
| **Accuracy** | 30 | Citations, data support, verifiability |
| **Purpose** | 20 | Educational/research vs. marketing/ads |
| **Timeliness** | 10 | Publication date, freshness |
| **Coverage** | 10 | Depth, completeness |

Only sources above threshold (default 60/100) contribute evidence. No domain whitelists — a personal blog with solid arguments scores just as well as a .edu page with fluff.

![Score detail](docs/screenshots/score-detail.png)

### 3. Self-evolving source memory

The real differentiator. The system doesn't just search — it remembers what worked.

After each run:
- Records how each source performed (scores, pass rate)
- Tiers sources automatically: **Elite → Trusted → Verified → Trial → Deprecated**
- Next time, prioritizes sources that historically score well on similar topics

In practice:
- Run 1: Normal search, already solid results
- Run 5: Source history kicks in, faster convergence
- Run 20: High-value sources accumulated, noticeably better output

Happens entirely in the background. No user action needed.

### 4. Transparent process

You can watch everything in real time:
- Agent reasoning (why it picked a search strategy)
- Exact queries used
- Which pages were visited
- Each page's score and why

![Search log](docs/screenshots/search-log.png)

### 5. Two-stage evaluation

Searching and scoring are separate steps:

- **Light pass**: Agent looks at search results, picks which links are worth visiting
- **Deep pass**: After visiting, LLM scores the page on all five dimensions before admitting any evidence

More efficient than scoring everything blindly.

### 6. Structured output

Results come as arguments + evidence, not a wall of text:

![Results](docs/screenshots/results.png)

Each piece of evidence includes:
- Summary
- Original source link (click to verify)
- Five-dimension score (expandable)
- Source domain

---

## Quick Start

### Requirements

- Python 3.10+
- LLM API key (OpenAI-compatible)
- Serper API key

### Setup

```bash
git clone https://github.com/HYC-hsy/deep-truth-search.git
cd deep-truth-search

# conda
conda create -n deep-truth-search python=3.10
conda activate deep-truth-search

# or venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### Configure

Edit `.env`. Two LLMs, different jobs:

| Purpose | Role | Recommended | Notes |
|---------|------|-------------|-------|
| **Main LLM** | Reasoning, decomposition, search strategy, extraction | claude-opus-4-6 | Use the strongest model you can afford |
| **Scoring LLM** (optional) | Quality scoring | gpt-4o | Simpler task, cheaper model works |

> Without a scoring LLM configured, the main LLM handles both. Works fine, just costs more.

```env
# Main LLM (required)
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL=claude-opus-4-6

# Scoring LLM (optional)
SCORING_LLM_API_KEY=sk-your-scoring-key
SCORING_LLM_BASE_URL=https://api.openai.com/v1
SCORING_LLM_MODEL=gpt-4o

# Search (required)
SEARCH_PROVIDER=serper
SEARCH_API_KEY=your-serper-key
```

[Serper](https://serper.dev/) gives 2500 free searches on signup — enough for serious use.

> Full config reference: [.env.example](.env.example)

### Run

```bash
python main.py
```

Opens `http://127.0.0.1:8888` automatically. Type a claim, watch it work.

> First search takes 3-10 minutes depending on complexity and network. Progress streams in real time.

---

## Architecture

```
Claim input
    |
    v
Main Agent (research controller)
    |-- Decompose claim into sub-angles
    |-- Load historical high-quality sources
    |-- Dispatch Search Agents in parallel
    |-- Evaluate coverage, decide if more searching needed
    |-- Select memory candidates -> update source memory
    |-- Assemble structured output
    |
    v
Search Agent x N (parallel)
    |-- Generate queries (multilingual, multi-angle)
    |-- Call search API
    |-- Decide which results to visit
    |-- Visit pages, extract content
    |-- Five-dimension scoring
    |-- Extract evidence
    |-- Decide if more searching needed
    |
    v
Structured output (arguments + evidence + scores + links)
```

### Core Modules

| Module | Path | What it does |
|--------|------|--------------|
| Main Agent | `agents/main_agent.py` | Research control loop |
| Search Agent | `agents/search_agent.py` | Single sub-claim deep search |
| Agent Loop | `agents/agent_loop.py` | Generic think-act-observe engine |
| Search Tool | `tools/search_tool.py` | Search API wrapper (Provider pattern) |
| Page Visitor | `tools/visit_tool.py` | Web/PDF content extraction |
| Scoring | `scoring/scoring.py` | LLM-based quality assessment |
| Source Memory | `memory/source_memory.py` | Five-tier ranking + promotion/demotion |
| Web UI | `ui/` | ChatGPT-style frontend (zero deps, ~30KB) |

---

## Tech Stack

- **Backend**: Python, FastAPI, SSE (real-time progress)
- **Frontend**: Vanilla HTML/CSS/JS (no framework, ~30KB)
- **LLM**: Any OpenAI-compatible API (Claude / GPT / DeepSeek / etc.)
- **Search**: Serper (Provider pattern, easy to add others)
- **Extraction**: trafilatura (HTML), pymupdf4llm (PDF)
- **Models**: Pydantic
- **Storage**: JSON (Repository pattern, extensible to SQLite)

---

## Configuration

Key settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `claude-opus-4-6` | Main LLM |
| `QUALITY_THRESHOLD` | `60` | Evidence quality threshold (0-100) |
| `MEMORY_THRESHOLD` | `75` | Source memory admission threshold |
| `MAX_PARALLEL_SEARCHES` | `3` | Parallel search agents |
| `MAIN_AGENT_MAX_TURNS` | `12` | Main Agent max turns |
| `SEARCH_AGENT_MAX_TURNS` | `8` | Search Agent max turns |
| `DISCLOSURE_WINDOW_SIZE` | `15` | Historical sources shown to agent |

Full reference: [.env.example](.env.example)

---

## Dark Mode

Follows system preference, or toggle manually top-right.

![Dark mode](docs/screenshots/dark-mode-results.png)

---

## Who is this for

- **Anyone with a claim** — type it in, get evidence from multiple angles
- **Students** — gather references fast
- **Researchers** — multi-source evidence collection
- **Content creators** — find reliable material for articles/videos
- **Anyone who needs evidence** — no research skills required

---

## Contributing

- Bug reports / feature requests → [Issues](https://github.com/HYC-hsy/deep-truth-search/issues)
- PRs welcome
- Add search adapters (implement `SearchProvider`)
- Improve scoring logic
- Improve UI/UX

---

## License

[MIT](LICENSE)

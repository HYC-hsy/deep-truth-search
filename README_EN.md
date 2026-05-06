# Deep Truth Search

**An open-source evidence search assistant that gets smarter with every use.**

> [中文版](README.md)

---

## What Is This

You have a claim. You want evidence to support it — or challenge it.

Regular search engines give you a pile of links. You open them one by one, judge quality yourself, and filter for useful content. It's slow and exhausting.

**Deep Truth Search does this for you.**

Enter a claim, and the system will:
1. Automatically decompose it into multiple research directions
2. Search in parallel, visiting real pages in depth
3. Evaluate each page's quality, filtering out low-quality sources
4. Return structured "claims + evidence," each with source links and scores

You only do the last step: **judge for yourself, from a large pool of high-quality evidence.**

![Home](docs/screenshots/home.png)

---

## Why This Project

### Problems with Existing Tools

| Problem | Details |
|---------|---------|
| Many results, few useful evidences | Search engines optimize for "finding information," not "finding evidence" |
| Good and bad sources mixed together | Users struggle to quickly judge if a page is credible |
| Every search starts from scratch | Systems don't remember which sources performed well before |
| No quality assessment | No tool helps you evaluate a source's authority or accuracy |

### Design Philosophy

> **Don't make judgments for the user. Just provide abundant high-quality evidence.**

This is not an "answer engine." It's an **evidence collection tool** — searches deep, searches wide, assesses quality, traces sources. The final judgment is always yours.

---

## Key Features

### 1. Purpose-Built for Evidence Finding

This isn't general search or AI chat. The system is optimized for one scenario: **you have a claim and need evidence for it.**

It automatically decomposes your claim into sub-directions (technical, applied, academic, policy, etc.) and searches them in parallel — far broader coverage than a single keyword search.

![Parallel Search](docs/screenshots/searching-progress.png)

### 2. Five-Dimensional Quality Scoring

Evidence isn't accepted just because it was found. The system uses an LLM to simulate expert reviewers, scoring independently across five dimensions:

| Dimension | Max Score | What It Evaluates |
|-----------|-----------|-------------------|
| **Authority** | 30 | Author credentials, institutional backing, expertise |
| **Accuracy** | 30 | Citations, data support, verifiability |
| **Purpose** | 20 | Education/research vs. marketing/advertising |
| **Timeliness** | 10 | Publication date, content freshness |
| **Coverage** | 10 | Depth of analysis, completeness |

Only pages scoring above the quality threshold (default 60/100) contribute evidence. The scoring logic doesn't rely on domain whitelists — it lets the LLM judge from content itself. A personal blog with rigorous argumentation can score just as high as a major outlet.

![Score Detail](docs/screenshots/score-detail.png)

### 3. Self-Evolving Source Memory

This is the project's biggest differentiator. The system doesn't just search — **it learns where to search.**

After each research task, the system:
- Records each source's performance (scores, pass rates)
- Automatically tiers sources: **Elite → Trusted → Verified → Trial → Deprecated**
- Prioritizes historically high-scoring sources for similar future topics

This means:
- **1st use**: Normal search, already good results
- **5th use**: System leverages historical sources, faster searches
- **20th use**: Enough high-value sources accumulated, noticeably more accurate

Users don't need to do anything. Self-evolution happens entirely in the background.

### 4. Fully Transparent Search Process

Not a black box. Click any search direction and a side panel shows in real-time:
- The Agent's thinking process (why it chose this search strategy)
- Specific search queries issued
- Which pages were visited
- Each page's score and reasoning

![Search Log](docs/screenshots/search-log.png)

### 5. Two-Stage Evaluation Architecture

Search and evaluation happen in two stages, saving resources while maintaining quality:

- **Light evaluation**: After search results return, the Agent autonomously decides which links are worth visiting
- **Heavy evaluation**: After visiting pages, the LLM scores them against all five dimensions to determine evidence pool entry

More efficient than "use everything found" or "deep-evaluate everything."

### 6. Structured Output

Results aren't a messy paragraph of summary. They're organized as "claims + evidence":

![Results](docs/screenshots/results.png)

Each piece of evidence includes:
- Evidence summary
- Original source link (clickable for tracing)
- Five-dimensional score (expandable for details)
- Source domain

---

## Quick Start

### Requirements

- Python 3.10+
- LLM API Key (instructions below)
- Search API Key (instructions below)

### Step 0: Clone the Project

```bash
git clone https://github.com/HYC-hsy/deep-truth-search.git
cd deep-truth-search
```

### Step 1: Create a Virtual Environment

Use conda or venv to create an isolated environment:

```bash
# Option A: conda (recommended)
conda create -n deep-truth-search python=3.10
conda activate deep-truth-search

# Option B: venv
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with the two required API keys:

This system uses two LLMs with different roles:

| Role | Purpose | Recommended Models | Notes |
|------|---------|-------------------|-------|
| **Main LLM** | Agent thinking, claim decomposition, search strategy, evidence extraction | Claude Opus / GPT-4o / DeepSeek-R1 or other strong models | Requires strong reasoning and planning — use the best model you can afford |
| **Scoring LLM** (optional) | Five-dimensional quality scoring | GPT-4o-mini / Claude Haiku or other lightweight models | Relatively simple task — a cheaper model works fine, saves cost |

> If you don't configure a separate scoring LLM, the system uses the main LLM for both roles — works fine, but costs more.

**Main LLM API Key** (required):

Any OpenAI-compatible API works:
- [OpenAI](https://platform.openai.com/api-keys) — recommended: `gpt-5.1` or stronger
- [Anthropic Claude](https://console.anthropic.com/) — recommended: `claude-sonnet-4-20250514` or `claude-opus-4-20250514`
- Other OpenAI-compatible APIs ([DeepSeek](https://platform.deepseek.com/), Qwen, etc.)

```env
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
```

**Scoring LLM API Key** (optional, recommended to save cost):

```env
SCORING_LLM_API_KEY=sk-your-scoring-api-key
SCORING_LLM_BASE_URL=https://api.openai.com/v1
SCORING_LLM_MODEL=gpt-4o-mini
```

**Search API Key** (required):

- [Serper](https://serper.dev/) — free tier gives 2,500 searches, enough for extensive use

```env
SEARCH_API_KEY=your-serper-api-key-here
```

> See [.env.example](.env.example) for all configuration options with comments.

### Step 4: Run

```bash
python main.py
```

Your browser will automatically open `http://127.0.0.1:8888`. Enter your claim and wait for evidence.

> **Note**: The first search typically takes 3-10 minutes depending on claim complexity and network conditions. You can watch real-time progress during the search.

---

## Architecture

```
User enters a claim
    |
    v
Main Agent (Global Research Controller)
    |-- Decompose into sub-claims (multi-angle coverage)
    |-- Load historical high-quality sources (disclosure window)
    |-- Dispatch Search Agents in parallel
    |-- Evaluate coverage, decide whether to supplement
    |-- Select memory candidates → write to long-term memory
    |-- Assemble structured output
    |
    v
Search Agent x N (parallel, acts as a high-level tool for Main Agent)
    |-- Generate search queries (multi-language, multi-angle)
    |-- Call search API for candidates
    |-- Agent decides which are worth visiting
    |-- Visit pages, extract structured content
    |-- Five-dimensional quality scoring
    |-- Extract evidence snippets
    |-- Decide whether to search more
    |
    v
Structured Output (claims + evidence + scores + links)
```

### Core Modules

| Module | Path | Role |
|--------|------|------|
| Main Agent | `agents/main_agent.py` | Global research control loop |
| Search Agent | `agents/search_agent.py` | Deep search executor per sub-claim |
| Agent Loop | `agents/agent_loop.py` | Generic think-act-observe loop engine |
| Search Tool | `tools/search_tool.py` | Search API wrapper (Provider pattern) |
| Visit Tool | `tools/visit_tool.py` | Web page / PDF content extraction |
| Scoring | `scoring/scoring.py` | LLM-driven quality assessment (no domain whitelists) |
| Source Memory | `memory/source_memory.py` | Five-tier classification + promotion/demotion + disclosure window |
| Web UI | `ui/` | ChatGPT-style frontend (zero framework deps) |

---

## Tech Stack

- **Backend**: Python, FastAPI, SSE (real-time search progress)
- **Frontend**: Vanilla HTML/CSS/JS (zero framework dependencies, ~30KB)
- **LLM**: OpenAI / Claude (unified interface, extensible to any OpenAI-compatible API)
- **Search**: Serper (Google Search API, Provider pattern for extensibility)
- **Page Extraction**: trafilatura (HTML), pymupdf4llm (PDF)
- **Data Models**: Pydantic
- **Storage**: JSON files (Repository pattern, upgradeable to SQLite)

---

## Configuration

All configuration is managed via `.env`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | LLM model name |
| `QUALITY_THRESHOLD` | `60` | Evidence quality threshold (0-100) |
| `MEMORY_THRESHOLD` | `75` | Source memory threshold (only sources above this are remembered) |
| `MAX_PARALLEL_SEARCHES` | `3` | Number of parallel searches |
| `MAIN_AGENT_MAX_TURNS` | `12` | Main Agent max turns |
| `SEARCH_AGENT_MAX_TURNS` | `8` | Search Agent max turns |
| `DISCLOSURE_WINDOW_SIZE` | `15` | Disclosure window size (historical sources shown to Agent) |

See [.env.example](.env.example) for full configuration with comments.

---

## Dark Mode

Supports light / dark themes. Follows system preference or toggle manually in the top-right corner.

![Dark Mode](docs/screenshots/dark-mode-results.png)

---

## Who Is This For

- **Anyone verifying a claim** — enter a claim, get multi-angle evidence
- **Students writing papers/reports** — quickly gather high-quality references
- **Teachers and researchers** — topic surveys, multi-source evidence collection
- **Content creators** — find reliable supporting material for articles and videos
- **Anyone who needs to "find evidence"** — no professional search skills required

---

## Contributing

Contributions welcome! You can:

- Submit bug reports or feature requests → [Issues](https://github.com/HYC-hsy/deep-truth-search/issues)
- Submit Pull Requests
- Add new search engine adapters (implement the `SearchProvider` interface)
- Improve scoring logic
- Improve UI/UX

---

## License

[MIT License](LICENSE)

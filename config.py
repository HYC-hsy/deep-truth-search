"""
Deep Truth Search — 统一配置管理

所有配置从 .env 文件加载，集中暴露为 Python 对象。
用法：from config import cfg
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 加载 .env ─────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _int(key: str, default: int = 0) -> int:
    return int(_get(key, str(default)))


def _float(key: str, default: float = 0.0) -> float:
    return float(_get(key, str(default)))


def _bool(key: str, default: bool = False) -> bool:
    return _get(key, str(default)).lower() in ("true", "1", "yes")


# ── LLM 配置 ─────────────────────────────────────────────────
class LLMConfig:
    provider: str    = _get("LLM_PROVIDER", "openai")
    api_key: str     = _get("LLM_API_KEY")
    base_url: str    = _get("LLM_BASE_URL", "https://api.openai.com/v1")
    model: str       = _get("LLM_MODEL", "gpt-4o")
    max_tokens: int  = _int("LLM_MAX_TOKENS", 4096)
    temperature: float = _float("LLM_TEMPERATURE", 0.3)
    timeout: int     = _int("LLM_TIMEOUT", 120)
    max_retries: int = _int("LLM_MAX_RETRIES", 2)


# ── 评分 LLM 配置（不填则 fallback 到主 LLM）─────────────────
class ScoringLLMConfig:
    api_key: str     = _get("SCORING_LLM_API_KEY")
    base_url: str    = _get("SCORING_LLM_BASE_URL")
    model: str       = _get("SCORING_LLM_MODEL")
    max_retries: int = _int("SCORING_LLM_MAX_RETRIES", 0)

    @classmethod
    def is_configured(cls) -> bool:
        return bool(cls.api_key and cls.model)


# ── 搜索 API 配置 ────────────────────────────────────────────
class SearchConfig:
    provider: str = _get("SEARCH_PROVIDER", "tavily")
    api_key: str  = _get("SEARCH_API_KEY")


# ── Web 服务配置 ──────────────────────────────────────────────
class WebConfig:
    host: str      = _get("WEB_HOST", "127.0.0.1")
    port: int      = _int("WEB_PORT", 8888)
    auto_open: bool = _bool("WEB_AUTO_OPEN", True)


# ── Agent 控制参数 ────────────────────────────────────────────
class AgentConfig:
    main_max_rounds: int     = _int("MAIN_AGENT_MAX_ROUNDS", 3)
    main_max_turns: int      = _int("MAIN_AGENT_MAX_TURNS", 12)
    search_budget: int       = _int("SEARCH_AGENT_BUDGET", 5)
    search_results_per_query: int = _int("SEARCH_RESULTS_PER_QUERY", 10)
    search_max_visits: int   = _int("SEARCH_AGENT_MAX_VISITS", 10)
    search_max_turns: int    = _int("SEARCH_AGENT_MAX_TURNS", 8)
    quality_threshold: int   = _int("QUALITY_THRESHOLD", 60)
    memory_threshold: int    = _int("MEMORY_THRESHOLD", 75)
    max_parallel_searches: int = _int("MAX_PARALLEL_SEARCHES", 3)


# ── HTTP 请求配置 ─────────────────────────────────────────────
class HTTPConfig:
    connect_timeout: int      = _int("HTTP_CONNECT_TIMEOUT", 10)
    read_timeout: int         = _int("HTTP_READ_TIMEOUT", 30)
    user_agent: str           = _get("HTTP_USER_AGENT", "DeepTruthSearch/1.0")
    rate_limit_per_domain: float = _float("HTTP_RATE_LIMIT_PER_DOMAIN", 1.0)
    search_api_interval: float   = _float("SEARCH_API_INTERVAL", 1.0)


# ── 记忆系统配置 ──────────────────────────────────────────────
class MemoryConfig:
    storage: str           = _get("MEMORY_STORAGE", "json")
    dir: str               = _get("MEMORY_DIR", "data/memory")
    disclosure_window: int = _int("DISCLOSURE_WINDOW_SIZE", 15)


# ── 统一入口 ──────────────────────────────────────────────────
class Config:
    llm: LLMConfig               = LLMConfig
    scoring_llm: ScoringLLMConfig = ScoringLLMConfig
    search: SearchConfig         = SearchConfig
    web: WebConfig       = WebConfig
    agent: AgentConfig   = AgentConfig
    http: HTTPConfig     = HTTPConfig
    memory: MemoryConfig = MemoryConfig

    # 项目根目录
    root: Path = Path(__file__).parent


cfg = Config()

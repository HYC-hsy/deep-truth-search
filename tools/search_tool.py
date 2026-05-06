"""
Deep Truth Search — 搜索工具

封装外部搜索 API，输入查询字符串，返回候选链接列表。
采用 SearchProvider 抽象接口，便于后续替换搜索引擎。

用法：
    from tools.search_tool import search

    results = await search("AI 2024 突破")
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx

from config import cfg
from models import SearchResult
from tools.http_client import get_client, get_search_rate_limiter

logger = logging.getLogger(__name__)


# ── 抽象接口 ──────────────────────────────────────────────────


class SearchProvider(ABC):
    """搜索引擎抽象接口，新增引擎只需实现此接口并注册"""

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        ...


# ── Serper 实现（Google Search）────────────────────────────────


class SerperProvider(SearchProvider):
    """基于 Serper.dev Google Search API 的搜索实现"""

    API_URL = "https://google.serper.dev/search"

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # 首次重试等待秒数

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not cfg.search.api_key:
            raise RuntimeError("SEARCH_API_KEY 未配置，请在 .env 中填写 Serper API Key")

        headers = {
            "X-API-KEY": cfg.search.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": max_results,
        }

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            await get_search_rate_limiter().wait()
            try:
                async with get_client() as client:
                    resp = await client.post(self.API_URL, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                break  # 成功，跳出重试循环
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Serper 搜索重试 %d/%d '%s': [%s] %s (%.1fs 后重试)",
                        attempt + 1, self.MAX_RETRIES, query[:40],
                        type(e).__name__, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        results: list[SearchResult] = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("link", ""),
            ))

        logger.info("Serper 搜索 '%s' 返回 %d 条结果", query[:50], len(results))
        return results


# ── Provider 注册表 ─────────────────────────────────────────────

_PROVIDERS: dict[str, type[SearchProvider]] = {
    "serper": SerperProvider,
}


def _get_provider() -> SearchProvider:
    provider_name = cfg.search.provider.lower()
    cls = _PROVIDERS.get(provider_name)
    if cls is None:
        raise RuntimeError(f"不支持的搜索提供商: {provider_name}，可选: {', '.join(_PROVIDERS)}")
    return cls()


# ── 公开接口 ─────────────────────────────────────────────────────


async def search(query: str, max_results: int = 10) -> list[SearchResult]:
    """执行搜索，返回候选链接列表。

    Args:
        query: 搜索查询字符串
        max_results: 最大返回条数

    Returns:
        List[SearchResult]，每条含 title、snippet、url
    """
    provider = _get_provider()
    return await provider.search(query, max_results=max_results)

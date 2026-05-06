"""
Deep Truth Search — 共享 HTTP Client

为 Search Tool 和 Visit Tool 提供统一的 HTTP Client 实例。
统一配置：User-Agent、请求超时、连接池、域名级速率控制、基础重试策略。

用法：
    from tools.http_client import get_client, fetch_url

    # 简单场景：带限速和重试的 GET
    resp = await fetch_url("https://example.com")

    # 自定义场景：直接使用 client
    async with get_client() as client:
        resp = await client.post(url, json=payload)
"""

import asyncio
import time
from collections import defaultdict

import httpx

from config import cfg


# ── 域名级速率控制 ────────────────────────────────────────────

class DomainRateLimiter:
    """按域名限速：同一域名两次请求之间至少间隔 min_interval 秒"""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def wait(self, domain: str) -> None:
        async with self._locks[domain]:
            now = time.monotonic()
            elapsed = now - self._last_request[domain]
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request[domain] = time.monotonic()


class APIRateLimiter:
    """搜索 API 全局速率控制。

    控制整体调用节奏，用于按 key 计费的搜索 API 等场景。
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def wait(self) -> None:
        """等待直到可以发起下一次 API 调用。"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()
            self._call_count += 1


# ── 模块级单例 ────────────────────────────────────────────────

_rate_limiter = DomainRateLimiter(min_interval=cfg.http.rate_limit_per_domain)
_search_rate_limiter = APIRateLimiter(min_interval=cfg.http.search_api_interval)


def get_search_rate_limiter() -> APIRateLimiter:
    """获取搜索 API 速率控制器。"""
    return _search_rate_limiter

_timeout = httpx.Timeout(
    connect=float(cfg.http.connect_timeout),
    read=float(cfg.http.read_timeout),
    write=10.0,
    pool=10.0,
)

_headers = {
    "User-Agent": cfg.http.user_agent,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_limits = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
)


def get_client(**overrides: object) -> httpx.AsyncClient:
    """创建配置好的 AsyncClient 实例。调用方应使用 async with 管理生命周期。"""
    kwargs: dict = {
        "timeout": _timeout,
        "headers": _headers,
        "limits": _limits,
        "follow_redirects": True,
    }
    kwargs.update(overrides)
    return httpx.AsyncClient(**kwargs)


async def fetch_url(url: str, *, max_retries: int = 2) -> httpx.Response:
    """带速率控制和重试的 URL 请求。

    - 按域名限速
    - 指数退避重试（仅重试 5xx / 超时）
    - 返回 httpx.Response，调用方自行判断 status_code
    """
    from urllib.parse import urlparse
    domain = urlparse(url).netloc

    await _rate_limiter.wait(domain)

    last_exc: Exception | None = None
    async with get_client() as client:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.get(url)
                if resp.status_code >= 500 and attempt < max_retries:
                    await asyncio.sleep(1.5 * (2 ** attempt))
                    continue
                return resp
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    await asyncio.sleep(1.5 * (2 ** attempt))
                    continue
                raise

    raise last_exc  # type: ignore[misc]

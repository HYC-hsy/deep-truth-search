"""
Deep Truth Search — Search Agent Handler

Search Agent 的 Agent Loop 处理器。
LLM 自主决定搜索词、选择访问页面、判断是否补搜、何时提交证据。

工具：
  - web_search(query) → 返回搜索结果列表
  - visit_page(url) → 访问页面 + 内部 LLM 提取证据 → 返回页面信息和证据摘要
  - submit_evidence() → 终止循环，返回累积的全部证据
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from agents.agent_loop import StepOutcome
from config import cfg
from models import EvidenceItem, PageContent, ScoreResult

logger = logging.getLogger(__name__)

# 跨子观点共享的页面缓存类型：url → (PageContent, ScoreResult) | None（访问失败）
PageCache = dict[str, tuple[PageContent, ScoreResult] | None]


# ── 工具 Schema ──────────────────────────────────────────────

SEARCH_AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索网络，获取与子观点相关的候选结果。支持批量搜索：传入多个搜索词可一次性并行搜索，结果按搜索词分组返回。",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索查询字符串数组。可以传入多个不同角度的搜索词，系统会并行执行。",
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit_page",
            "description": "访问 URL 并提取与子观点相关的证据。支持批量访问：传入多个 URL 可一次性并行访问，系统会自动评分和提取证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要访问的 URL 数组。可以一次传入多个 URL，系统会并行处理。",
                    }
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_evidence",
            "description": "提交收集到的证据并结束搜索。当已收集到足够的证据，或已穷尽搜索途径时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ── 证据提取 Prompt ──────────────────────────────────────────

EXTRACT_EVIDENCE_PROMPT = """\
你是一个证据提取专家。从网页内容中提取直接支持子观点的证据。

## 提取原则

1. 只提取直接支持子观点的具体事实、数据或引用，泛泛的行业背景和趋势概述不算证据
2. 每条证据应完整独立，描述同一件事的相邻内容应整合为一条，不要拆成碎片
3. 如果页面内容与子观点无关或只是间接相关，返回空列表
4. 证据用中文表达（英文原文翻译为中文并保留关键术语）
"""


# ── Handler 实现 ─────────────────────────────────────────────


class SearchAgentHandler:
    """Search Agent 的 Agent Loop Handler。

    内部维护已收集的证据列表和搜索统计。
    """

    def __init__(
        self,
        subclaim: str,
        topic_context: str,
        search_budget: int = 5,
        page_cache: PageCache | None = None,
        preferred_sources: list[str] | None = None,
    ):
        self.subclaim = subclaim
        self.topic_context = topic_context
        self.search_budget = search_budget
        self._page_cache: PageCache = page_cache if page_cache is not None else {}
        self._preferred_sources: list[str] = preferred_sources or []

        # 内部状态
        self._evidence_items: list[EvidenceItem] = []
        self._searches_done: int = 0
        self._pages_visited: int = 0
        self._visited_urls: set[str] = set()
        self._accepted_sources: list[str] = []
        self._rejected_sources: list[str] = []

    def get_system_prompt(self) -> str:
        today = time.strftime('%Y-%m-%d %a')
        return f"""\
你是一个专注于证据搜索的执行 Agent。

当前日期：{today}

## 当前任务

为以下子观点搜索高质量证据：
「{self.subclaim}」

主题上下文：{self.topic_context}

## 工作策略

1. 用 web_search 搜索证据。可以一次提交多个不同角度的搜索词，系统会并行执行
2. 浏览搜索结果，选择有价值的页面访问（visit_page）
   - 优先选择权威来源：学术机构、官方网站、权威媒体
   - 跳过明显无关或低质量的结果（论坛灌水、营销页面等）
   - PDF 链接也可以访问（系统会自动提取 PDF 内容）
3. 每次访问页面后，系统会自动提取证据
4. 如果当前搜索结果不够好，换一个角度或关键词重新搜索
5. 当收集到足够证据（至少 2-3 条有价值的），或已尝试多个搜索角度后，调用 submit_evidence 结束

## 约束

- 搜索次数有限，合理规划搜索策略
- 访问你认为有价值且相关的页面，不要盲目全部访问
- 证据要具体（有数据、有事实），不要笼统描述
- 如果某个方向搜不到好结果，及时换方向
{self._format_preferred_sources_section()}"""

    def _format_preferred_sources_section(self) -> str:
        """将优先来源列表格式化为 system prompt 片段（P5-I2）。"""
        if not self._preferred_sources:
            return ""

        domains_str = "、".join(self._preferred_sources[:10])
        return f"""

## 优先来源（系统推荐）

以下域名在过往搜索中表现优秀，搜索时可优先关注：
{domains_str}

提示：首轮搜索可尝试在这些来源中搜索（如 site:domain.com 关键词），
后续轮次进行正常广域搜索以发现新来源。
"""

    def get_tools_schema(self) -> list[dict]:
        return SEARCH_AGENT_TOOLS

    def on_turn_start(self, turn: int, max_turns: int) -> Optional[str]:
        remaining = max_turns - turn
        if remaining == 1:
            evidence_count = len(self._evidence_items)
            return (
                f"这是最后一轮。当前已收集 {evidence_count} 条证据。"
                "请立即调用 submit_evidence 提交已有证据。"
            )
        if remaining == 2:
            evidence_count = len(self._evidence_items)
            return (
                f"还剩 2 轮。当前已收集 {evidence_count} 条证据。"
                "请尽快完成搜索并调用 submit_evidence。"
            )
        return None

    def on_max_turns(self) -> Any:
        logger.warning("Search Agent 超时，返回已收集的 %d 条证据", len(self._evidence_items))
        return self._evidence_items

    async def dispatch(self, tool_name: str, args: dict) -> StepOutcome:
        method = getattr(self, f"do_{tool_name}", None)
        if method is None:
            return StepOutcome(data={"error": f"未知工具: {tool_name}"})
        return await method(args)

    # ── 工具实现 ──────────────────────────────────────────

    def _is_preferred_domain(self, url: str) -> bool:
        """检查 URL 是否来自优先来源域名。"""
        if not self._preferred_sources:
            return False
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            return any(pref in domain for pref in self._preferred_sources)
        except Exception:
            return False

    async def do_web_search(self, args: dict) -> StepOutcome:
        from tools.search_tool import search
        import asyncio

        # 兼容单个 query（旧格式）和 queries 数组（新格式）
        queries = args.get("queries") or []
        if not queries and args.get("query"):
            queries = [args["query"]]
        if not queries:
            return StepOutcome(data={"error": "未提供搜索词"})

        is_first_search = self._searches_done == 0
        max_results = cfg.agent.search_results_per_query

        # P5-I3：首次搜索 + 有优先来源 → 自动补充一次 site-scoped 搜索
        site_queries = []
        if (is_first_search
                and self._preferred_sources
                and not any("site:" in q.lower() for q in queries)):
            site_domains = self._preferred_sources[:3]
            site_suffix = " " + " OR ".join(f"site:{d}" for d in site_domains)
            site_queries = [queries[0] + site_suffix]

        # 并行执行所有搜索
        all_queries = site_queries + queries

        async def _do_one_search(q: str, is_site: bool = False) -> tuple[str, list, bool]:
            try:
                mr = 5 if is_site else max_results
                results = await search(q, max_results=mr)
                return q, results, is_site
            except Exception as e:
                logger.error("搜索失败 '%s': [%s] %r", q[:50], type(e).__name__, e)
                return q, [], is_site

        tasks = []
        for sq in site_queries:
            tasks.append(_do_one_search(sq, is_site=True))
        for q in queries:
            tasks.append(_do_one_search(q, is_site=False))

        search_results = await asyncio.gather(*tasks)

        # 统计和日志
        self._searches_done += len(queries)
        for q, results, is_site in search_results:
            if is_site:
                logger.info("优先来源搜索补充: %d 条结果 (query: %s)", len(results), q[:80])
            else:
                logger.info("Search Agent 搜索 [%d]: %s → %d 条结果",
                            self._searches_done, q[:60], len(results))

        # 合并所有结果，去重
        seen_urls: set[str] = set()
        merged = []
        # 优先来源结果在前
        for q, results, is_site in search_results:
            if is_site:
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        merged.append(r)
        # 然后是正常搜索结果
        for q, results, is_site in search_results:
            if not is_site:
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        merged.append(r)

        formatted = []
        for i, r in enumerate(merged, 1):
            entry = {
                "index": i,
                "title": r.title,
                "snippet": r.snippet,
                "url": r.url,
            }
            if self._is_preferred_domain(r.url):
                entry["preferred"] = True
            formatted.append(entry)

        preferred_count = sum(1 for f in formatted if f.get("preferred"))
        return StepOutcome(data={
            "queries": queries,
            "result_count": len(formatted),
            "results": formatted,
            **({"preferred_hits": preferred_count} if preferred_count else {}),
        })

    async def do_visit_page(self, args: dict) -> StepOutcome:
        """支持单个 URL 或批量 URL 并行访问。"""
        import asyncio

        # 兼容旧格式 url: string 和新格式 urls: array
        urls = args.get("urls") or []
        if not urls and args.get("url"):
            urls = [args["url"]]
        if not urls:
            return StepOutcome(data={"error": "未提供 URL"})

        # 预先注册所有 URL，防止并行时重复（asyncio 单线程，这里没有 await，安全）
        to_visit = []
        skipped = []
        for url in urls:
            if url in self._visited_urls:
                skipped.append({"url": url, "status": "skipped", "reason": "已访问过"})
            else:
                self._visited_urls.add(url)
                self._pages_visited += 1
                to_visit.append(url)

        if not to_visit:
            return StepOutcome(data={"results": skipped})

        # 并行访问所有页面
        logger.info("Search Agent 并行访问 %d 个页面", len(to_visit))
        results = await asyncio.gather(*[self._visit_one_url(url) for url in to_visit])

        all_results = skipped + list(results)
        # 如果只有一个 URL，直接返回单个结果（保持简洁）
        if len(urls) == 1:
            return StepOutcome(data=all_results[0])
        return StepOutcome(data={"results": all_results, "total_evidence_so_far": len(self._evidence_items)})

    async def _visit_one_url(self, url: str) -> dict:
        """访问单个 URL：fetch → 评分 → 提取证据。返回结果字典。"""
        from tools.visit_tool import visit
        from scoring.scoring import score_page

        # 检查跨子观点共享缓存
        if url in self._page_cache:
            cached = self._page_cache[url]
            if cached is None:
                logger.info("Search Agent 命中失败缓存，跳过: %s", url[:60])
                return {"url": url, "status": "failed", "reason": "该页面之前访问失败，已跳过"}
            page, score_result = cached
            logger.info("Search Agent 命中缓存: %s (%.0f/100)", page.domain, score_result.total_score)
        else:
            logger.info("Search Agent 访问页面: %s", url[:80])

            page = await visit(url)
            if page is None or not page.body_text or len(page.body_text.strip()) < 50:
                self._page_cache[url] = None
                self._record_failure(url)
                return {"url": url, "status": "failed", "reason": "页面为空或无法访问"}

            score_result = await score_page(page, self.subclaim)
            self._page_cache[url] = (page, score_result)

        if not score_result.passes_threshold:
            self._rejected_sources.append(url)
            dim_summary = ", ".join(
                f"{d.name}={d.score:.0f}/{d.max_score:.0f}" for d in score_result.dimensions
            )
            logger.info("页面未达标，跳过: %s (%.0f/100)", page.domain, score_result.total_score)
            return {
                "url": page.url,
                "title": page.title,
                "domain": page.domain,
                "status": "rejected",
                "reason": f"页面质量评分 {score_result.total_score:.0f}/100 未达到阈值 {cfg.agent.quality_threshold}",
                "score": score_result.total_score,
                "dimensions": dim_summary,
            }

        self._accepted_sources.append(url)

        # 达标页面：提取证据
        extracted = await self._extract_evidence(page.title, page.url, page.domain, page.body_text)

        for item in extracted:
            item.score = score_result

        self._evidence_items.extend(extracted)

        dim_summary = ", ".join(
            f"{d.name}={d.score:.0f}/{d.max_score:.0f}" for d in score_result.dimensions
        )
        return {
            "url": page.url,
            "title": page.title,
            "author": page.author or "未知",
            "date": page.date or "未知",
            "domain": page.domain,
            "score": score_result.total_score,
            "dimensions": dim_summary,
            "status": "accepted",
            "evidence_extracted": len(extracted),
            "evidence_summaries": [e.evidence_text[:80] for e in extracted],
            "body_preview": page.body_text[:100] if page.body_text else "",
        }

    async def do_submit_evidence(self, _args: dict) -> StepOutcome:
        logger.info("Search Agent 提交 %d 条证据", len(self._evidence_items))
        return StepOutcome(data=self._evidence_items, should_exit=True)

    # ── 内部辅助：失效记录 ──────────────────────────────

    def _record_failure(self, url: str) -> None:
        """对已在记忆中的来源记录访问失效（P4-H5，静默失败）"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if not domain:
                return
            from memory.source_memory import record_source_failure
            record_source_failure(domain=domain, reason="empty_or_inaccessible")
        except Exception as e:
            logger.warning("失效记录写入失败（不影响搜索）: %s", e)

    # ── 内部辅助：LLM 证据提取 ───────────────────────────

    async def _extract_evidence(
        self,
        title: str,
        url: str,
        domain: str,
        body_text: str,
    ) -> list[EvidenceItem]:
        """用 LLM 从页面正文中提取与子观点相关的证据。

        使用 Anthropic tool_use 强制结构化输出，避免文本 JSON 解析问题。
        """
        from llm.llm_client import call_llm_structured, _is_claude_model, call_llm_json
        from pydantic import BaseModel, Field

        body_preview = body_text[:3000]

        user_message = (
            f"子观点：{self.subclaim}\n\n"
            f"页面标题：{title}\n"
            f"页面来源：{url}\n\n"
            f"页面正文（部分）：\n{body_preview}"
        )

        try:
            if _is_claude_model():
                # Claude：使用 tool_use 保证 JSON 完整性（避免文本输出被代理截断）
                tool_schema = {
                    "type": "object",
                    "properties": {
                        "evidences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "证据摘要文本列表，每个字符串是一条独立的证据",
                        }
                    },
                    "required": ["evidences"],
                }
                raw = await call_llm_structured(
                    system_prompt=EXTRACT_EVIDENCE_PROMPT,
                    user_message=user_message,
                    tool_name="submit_evidences",
                    tool_description="提交从页面中提取到的证据列表",
                    tool_schema=tool_schema,
                    temperature=0.2,
                    max_tokens=8192,
                )
                evidence_texts = raw.get("evidences", []) if isinstance(raw, dict) else []
                if isinstance(evidence_texts, str):
                    import json as _json
                    try:
                        evidence_texts = _json.loads(evidence_texts)
                    except Exception:
                        evidence_texts = [evidence_texts]
            else:
                class ExtractedEvidence(BaseModel):
                    evidence_text: str = Field(description="证据摘要文本")
                class ExtractionResult(BaseModel):
                    evidences: list[ExtractedEvidence] = Field(default_factory=list)

                result = await call_llm_json(
                    system_prompt=EXTRACT_EVIDENCE_PROMPT,
                    user_message=user_message,
                    response_model=ExtractionResult,
                    temperature=0.2,
                    max_tokens=8192,
                )
                evidence_texts = [ev.evidence_text for ev in result.evidences]
        except Exception as e:
            logger.warning("LLM 证据提取失败 (%s): %s %s", url[:60], type(e).__name__, e)
            return []

        logger.info("证据提取 [%s] %s: %d 条", domain, title[:30], len(evidence_texts))

        items: list[EvidenceItem] = []
        for text in evidence_texts:
            if isinstance(text, str) and text.strip():
                items.append(EvidenceItem(
                    claim=self.subclaim,
                    evidence_text=text.strip(),
                    source_url=url,
                    source_title=title,
                    source_domain=domain,
                ))
            elif isinstance(text, dict) and text.get("evidence_text", "").strip():
                items.append(EvidenceItem(
                    claim=self.subclaim,
                    evidence_text=text["evidence_text"].strip(),
                    source_url=url,
                    source_title=title,
                    source_domain=domain,
                ))

        return items

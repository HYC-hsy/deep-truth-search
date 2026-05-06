"""
Deep Truth Search — Search Agent

局部自治的深搜执行器。对 Main Agent 来说是一个高级工具（Facade 模式）。

当前实现：Agent Loop 版本。
内部 LLM 自主决定搜索词、选择访问页面、判断是否补搜、何时提交证据。

用法（Main Agent 调用）：
    from agents.search_agent import search_evidence, search_evidence_loop

    # Agent Loop 版本（推荐）
    evidence_items = await search_evidence_loop(subclaim="...", topic_context="...")

    # 兼容 Facade 版本
    result = await search_evidence(SearchEvidenceInput(subclaim="...", topic_context="..."))
"""

from __future__ import annotations

import logging

from config import cfg
from models import (
    EvidenceItem,
    SearchEvidenceInput,
    SearchEvidenceResult,
)

logger = logging.getLogger(__name__)


async def search_evidence_loop(
    subclaim: str,
    topic_context: str,
    search_budget: int | None = None,
    page_cache: dict | None = None,
    preferred_sources: list[str] | None = None,
    on_step=None,
) -> tuple[list[EvidenceItem], list[str], list[str]]:
    """运行 Search Agent Loop，返回 (证据列表, 通过来源, 拒绝来源)。

    这是 Search Agent 的核心入口。内部启动 Agent Loop，
    LLM 自主决定搜索、访问、提取证据的流程。
    页面访问后会进行五维评分，未达标的页面跳过证据提取。

    Args:
        page_cache: 跨子观点共享的页面缓存，避免重复下载和评分。
        preferred_sources: 披露窗口中的优先来源域名列表（P5-I2）。
    """
    from agents.agent_loop import run_agent_loop
    from agents.search_handler import SearchAgentHandler

    budget = search_budget or cfg.agent.search_budget

    handler = SearchAgentHandler(
        subclaim=subclaim,
        topic_context=topic_context,
        search_budget=budget,
        page_cache=page_cache,
        preferred_sources=preferred_sources or [],
    )

    result = await run_agent_loop(
        handler=handler,
        user_message=f"请为以下子观点搜索证据：\n{subclaim}\n\n主题上下文：{topic_context}",
        max_turns=cfg.agent.search_max_turns,
        on_step=on_step,
    )

    evidence = result if isinstance(result, list) else []
    return evidence, handler._accepted_sources, handler._rejected_sources


async def search_evidence(input: SearchEvidenceInput) -> SearchEvidenceResult:
    """Search Agent Facade — 兼容旧接口。

    内部委托给 search_evidence_loop。
    """
    logger.info("Search Agent 开始执行: subclaim='%s'", input.subclaim[:60])

    evidence_items, accepted, rejected = await search_evidence_loop(
        subclaim=input.subclaim,
        topic_context=input.topic_context,
        search_budget=input.search_budget,
    )

    summary = (
        f"搜索预算: {input.search_budget} | "
        f"通过来源: {len(accepted)} | "
        f"拒绝来源: {len(rejected)} | "
        f"证据数量: {len(evidence_items)}"
    )

    logger.info("Search Agent 执行完成: %s", summary)

    return SearchEvidenceResult(
        subclaim=input.subclaim,
        evidence_items=evidence_items,
        accepted_sources=accepted,
        rejected_sources=rejected,
        search_trace=[],
        coverage_notes=f"Agent Loop 搜索完成，共 {len(evidence_items)} 条证据（{len(rejected)} 个来源因评分未达标被过滤）",
        execution_summary=summary,
    )

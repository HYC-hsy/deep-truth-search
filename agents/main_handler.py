"""
Deep Truth Search — Main Agent Handler

Main Agent 的 Agent Loop 处理器。
LLM 自主拆解子观点、决定搜索方向、判断覆盖度、组织最终结果。

工具：
  - batch_search(subclaims) → 并行启动多个 Search Agent Loop，返回全部证据摘要
  - submit_results(claims) → 终止循环，组装 ResearchOutput
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional, Union

from agents.agent_loop import StepOutcome
from config import cfg
from models import ClaimResult, DisclosureWindow, EvidenceItem, ResearchOutput

logger = logging.getLogger(__name__)

StatusCallback = Optional[Callable[[Union[str, dict]], None]]


# ── 工具 Schema ──────────────────────────────────────────────

MAIN_AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "batch_search",
            "description": "同时搜索多个子观点方向。将所有要搜索的子观点一次性提交，系统会并行执行深度搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subclaims": {
                        "type": "array",
                        "description": "子观点列表，每个包含要搜索的子观点和主题上下文",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subclaim": {
                                    "type": "string",
                                    "description": "要搜索证据的子观点（具体、可搜索的陈述句）",
                                },
                                "topic_context": {
                                    "type": "string",
                                    "description": "简短的主题上下文，帮助聚焦搜索",
                                },
                            },
                            "required": ["subclaim"],
                        },
                    },
                },
                "required": ["subclaims"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": "提交最终研究结果并结束。将收集到的证据按论点组织后调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "description": "论点列表，每个论点包含标题和关联的子观点关键词",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_title": {
                                    "type": "string",
                                    "description": "论点标题（简短）",
                                },
                                "subclaim_keys": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "该论点关联的子观点文本（从 batch_search 返回的子观点中选取）",
                                },
                            },
                            "required": ["claim_title", "subclaim_keys"],
                        },
                    },
                },
                "required": ["claims"],
            },
        },
    },
]


# ── Handler 实现 ─────────────────────────────────────────────


class MainAgentHandler:
    """Main Agent 的 Agent Loop Handler。

    内部维护全局证据池和搜索计数。
    """

    def __init__(self, query: str, on_status: StatusCallback = None):
        self.query = query
        self.on_status = on_status

        # 内部状态
        self._all_evidence: list[EvidenceItem] = []
        self._search_count: int = 0
        self._subclaim_evidence_map: dict[str, list[int]] = {}  # subclaim → evidence indices
        self._page_cache: dict = {}  # 跨子观点共享的页面缓存：url → (PageContent, ScoreResult)
        self._disclosure_window: DisclosureWindow | None = None  # 披露窗口（Phase 5）
        self._evidence_lock = asyncio.Lock()  # 保护并发时证据索引正确性

    def _load_disclosure_window(self) -> DisclosureWindow | None:
        """加载披露窗口：根据用户观点主题，检索历史优质来源（P5-I2）。"""
        try:
            from memory.source_memory import get_disclosure_window
            window = get_disclosure_window(self.query)
            if window.total > 0:
                self._disclosure_window = window
                return window
        except Exception as e:
            logger.warning("披露窗口加载失败（不影响搜索）: %s", e)
        return None

    def _format_disclosure_section(self) -> str:
        """将披露窗口格式化为 system prompt 的一段文本（P5-I2）。"""
        window = self._disclosure_window
        if not window or window.total == 0:
            return ""

        lines = ["\n## 历史优质来源（系统自进化记忆）\n"]
        lines.append(
            f"系统已从历史搜索中积累了 {window.total} 个与当前主题相关的优质来源。"
            "搜索时可优先关注这些域名，它们在过往研究中表现良好：\n"
        )
        for src in window.sources:
            level_label = {"elite": "精英", "trusted": "可信", "verified": "已验证", "trial": "试用"}.get(
                src.current_level.value, src.current_level.value
            )
            lines.append(f"- {src.domain}（{level_label}，均分 {src.avg_score:.0f}）")

        lines.append(
            "\n注意：这些来源仅供参考，不要局限于此。"
            "仍需广泛搜索以发现新的高质量来源。\n"
        )
        return "\n".join(lines)

    def get_system_prompt(self) -> str:
        max_searches = cfg.agent.main_max_rounds * 3
        today = time.strftime('%Y-%m-%d %a')

        # 加载披露窗口
        self._load_disclosure_window()
        disclosure_section = self._format_disclosure_section()

        return f"""\
你是一个研究主管 Agent，负责围绕用户观点寻找充分的证据支撑。

当前日期：{today}
当用户提到"今年"时，指的是 {time.strftime('%Y')} 年。搜索时请使用正确的年份。

## 当前任务

为以下观点寻找数量足够多、质量足够高的证据：
「{self.query}」

忠于用户的原始观点。不改写、不弱化、不替换用户的表述。你的价值是为这个观点找到最有力的证据。

## 工作策略

1. **分析观点**：先理解观点要证明什么，再设计搜索方向
   - 识别观点中的关键主张词，思考每个词对证据的要求（例如："最有效"需要与替代方案横向比较；"不可逆"需要论证锁定机制；"被低估"需要解释低估的原因和证据）
   - 思考证明该观点最有力的论证结构是什么（例如：因果链、排除法、反事实分析、历史演进、多层证据金字塔等），然后按这个结构设计搜索方向
2. **批量搜索**：将搜索方向一次性提交给 batch_search，系统会并行搜索
   - 子观点应是明确的研究方向，不是搜索关键词的堆砌
   - 方向之间应有层次和逻辑关系，而非简单平铺罗列
   - 可以包含一个探索局限性或边界条件的方向，放在最后
   - **一次调用 batch_search 提交所有方向，不要分多次调用**
3. **评估覆盖度**：batch_search 返回全部结果后，查看所有方向的证据，判断：
   - 哪些方向证据充足？
   - 哪些方向失败或证据不足，需要补搜？
   - 是否有遗漏的重要角度？
4. **补充搜索**（可选）：对不足的方向再调用一次 batch_search 补搜
5. **组织输出**：证据充足后，调用 submit_results 按论点组织最终结果

## 约束

- 总搜索方向不超过 15 个
- 首次 batch_search 应包含 3-10 个子观点方向
- 最终输出按「论点 + 论据」结构组织
- 论点标题简短（一句话）
- 不替用户做最终判断，只提供证据
- 所有输出用中文

## 证据关联

每次 batch_search 返回后，会告诉你每个子观点找到了多少条证据。
在 submit_results 时，用子观点的原文作为 subclaim_keys 将证据关联到论点。
一个论点可以关联多个子观点，一个子观点也可以被多个论点引用。
{disclosure_section}"""

    def get_tools_schema(self) -> list[dict]:
        return MAIN_AGENT_TOOLS

    def on_turn_start(self, turn: int, max_turns: int) -> Optional[str]:
        remaining = max_turns - turn
        if remaining == 1:
            return (
                f"这是最后一轮。当前共 {len(self._all_evidence)} 条证据。"
                "你必须立即调用 submit_results 提交已有成果。"
            )
        if remaining == 2:
            return (
                f"还剩 2 轮。当前共 {len(self._all_evidence)} 条证据。"
                "请评估是否需要最后一次补搜，然后准备调用 submit_results。"
            )
        # 首次 batch_search 返回后，引导评估覆盖度
        if self._search_count > 0 and turn <= 3:
            failed = [sc for sc, indices in self._subclaim_evidence_map.items() if not indices]
            weak = [sc for sc, indices in self._subclaim_evidence_map.items() if 0 < len(indices) <= 5]
            if failed or weak:
                parts = [f"当前共 {len(self._all_evidence)} 条证据，还剩 {remaining} 轮。"]
                if failed:
                    parts.append(f"以下方向搜索失败或无证据：{[s[:30] for s in failed]}")
                if weak:
                    parts.append(f"以下方向证据较少（≤5条）：{[s[:30] for s in weak]}")
                parts.append("请评估是否需要补搜这些方向，或换个角度重新搜索。也可以直接提交结果。")
                return " ".join(parts)
        return None

    def on_max_turns(self) -> Any:
        """超时兜底：用已有证据自动组装结果。"""
        logger.warning("Main Agent 超时，自动组装 %d 条证据", len(self._all_evidence))
        return self._build_fallback_output()

    async def dispatch(self, tool_name: str, args: dict) -> StepOutcome:
        # 向后兼容：LLM 若仍调用旧接口，包装为 batch_search
        if tool_name == "search_evidence":
            return await self.do_batch_search({"subclaims": [args]})
        method = getattr(self, f"do_{tool_name}", None)
        if method is None:
            return StepOutcome(data={"error": f"未知工具: {tool_name}"})
        return await method(args)

    # ── 工具实现 ──────────────────────────────────────────

    async def do_batch_search(self, args: dict) -> StepOutcome:
        """并行搜索多个子观点方向。"""
        from agents.search_agent import search_evidence_loop

        subclaims_list = args.get("subclaims", [])
        if not subclaims_list:
            return StepOutcome(data={"error": "subclaims 不能为空", "results": []})

        sem = asyncio.Semaphore(cfg.agent.max_parallel_searches)
        preferred_sources = self._disclosure_window.domains if self._disclosure_window else []

        logger.info("Main Agent 发起批量搜索: %d 个子观点 (并发度 %d)",
                     len(subclaims_list), cfg.agent.max_parallel_searches)

        # 通知前端：本批次所有子观点
        if self.on_status:
            self.on_status({
                "type": "batch_start",
                "subclaims": [item.get("subclaim", "")[:50] for item in subclaims_list],
                "total": len(subclaims_list),
            })

        async def _search_one(index: int, item: dict) -> dict:
            subclaim = item.get("subclaim", "")
            topic_context = item.get("topic_context", "")

            # 构造带 index 的步骤级回调，推送 search_log SSE 事件
            def _on_step(event_type: str, content: str) -> None:
                if self.on_status:
                    self.on_status({
                        "type": "search_log",
                        "index": index,
                        "event": event_type,
                        "content": content,
                    })

            async with sem:
                self._search_count += 1
                count = self._search_count
                logger.info("Main Agent 发起搜索 [%d]: %s", count, subclaim[:60])

                if self.on_status:
                    self.on_status({
                        "type": "search_start",
                        "index": index,
                        "subclaim": subclaim[:50],
                    })

                try:
                    evidence_items, accepted_sources, rejected_sources = await search_evidence_loop(
                        subclaim=subclaim,
                        topic_context=topic_context,
                        page_cache=self._page_cache,
                        preferred_sources=preferred_sources,
                        on_step=_on_step,
                    )
                except Exception as e:
                    logger.error("Search Agent Loop 执行失败 [%s]: %s", subclaim[:40], e)
                    if self.on_status:
                        self.on_status({
                            "type": "search_done",
                            "index": index,
                            "subclaim": subclaim[:50],
                            "evidence_found": 0,
                            "error": str(e),
                        })
                    return {"subclaim": subclaim, "error": str(e), "evidence_found": 0}

                # 将来源写入长期记忆
                self._record_sources(evidence_items, accepted_sources, rejected_sources, topic_context)

                # 用锁保护证据索引的正确性
                async with self._evidence_lock:
                    start_idx = len(self._all_evidence)
                    self._all_evidence.extend(evidence_items)
                    indices = list(range(start_idx, start_idx + len(evidence_items)))
                    self._subclaim_evidence_map[subclaim] = indices

                logger.info("搜索完成 [%s]: %d 条证据", subclaim[:40], len(evidence_items))

                if self.on_status:
                    self.on_status({
                        "type": "search_done",
                        "index": index,
                        "subclaim": subclaim[:50],
                        "evidence_found": len(evidence_items),
                    })

                return {
                    "subclaim": subclaim,
                    "evidence_found": len(evidence_items),
                    "sources_accepted": len(accepted_sources),
                    "sources_rejected": len(rejected_sources),
                    "evidence_summaries": [
                        {
                            "text": e.evidence_text[:100],
                            "source": e.source_url[:60],
                            "score": e.score.total_score if e.score else None,
                        }
                        for e in evidence_items
                    ],
                }

        results = await asyncio.gather(
            *[_search_one(i, item) for i, item in enumerate(subclaims_list)]
        )

        successful = sum(1 for r in results if "error" not in r)
        failed = len(results) - successful

        logger.info("批量搜索完成: %d 成功, %d 失败, 累计 %d 条证据",
                     successful, failed, len(self._all_evidence))

        return StepOutcome(data={
            "results": list(results),
            "successful": successful,
            "failed": failed,
            "total_evidence_accumulated": len(self._all_evidence),
            "total_searches_done": self._search_count,
        })

    def _match_subclaim_key(self, key: str) -> list[int]:
        """用子观点关键词从 _subclaim_evidence_map 中查找证据索引。

        匹配策略：精确匹配 → 包含匹配（key 是 map_key 的子串，或反过来）。
        """
        # 精确匹配
        if key in self._subclaim_evidence_map:
            return self._subclaim_evidence_map[key]

        # 包含匹配
        for map_key, indices in self._subclaim_evidence_map.items():
            if key in map_key or map_key in key:
                return indices

        return []

    async def do_submit_results(self, args: dict) -> StepOutcome:
        claims_raw = args.get("claims", [])

        logger.info("submit_results 收到 %d 个论点, 证据池总量 %d, 子观点映射 %d 条",
                     len(claims_raw), len(self._all_evidence),
                     len(self._subclaim_evidence_map))

        claims = []

        for c in claims_raw:
            title = c.get("claim_title", "")
            subclaim_keys = c.get("subclaim_keys", [])
            # 兼容旧格式：如果 LLM 仍传 evidence_indices，走旧逻辑
            old_indices = c.get("evidence_indices", [])
            if old_indices and not subclaim_keys:
                logger.warning("论点 '%s' 使用了旧的 evidence_indices 格式，尝试兼容",
                               title[:30])
                evidences = [self._all_evidence[i]
                             for i in old_indices
                             if 0 <= i < len(self._all_evidence)]
            else:
                # 通过子观点名称查找证据（允许同一证据出现在多个论点）
                evidences = []
                seen_url_texts: set[tuple[str, str]] = set()  # (URL, evidence_text) 去重
                matched_keys = []
                unmatched_keys = []

                for key in subclaim_keys:
                    indices = self._match_subclaim_key(key)
                    if indices:
                        matched_keys.append(key[:30])
                        for i in indices:
                            if 0 <= i < len(self._all_evidence):
                                ev = self._all_evidence[i]
                                dedup_key = (ev.source_url, ev.evidence_text)
                                if dedup_key not in seen_url_texts:
                                    evidences.append(ev)
                                    seen_url_texts.add(dedup_key)
                    else:
                        unmatched_keys.append(key[:30])

                if unmatched_keys:
                    logger.warning("  论点 '%s': %d 个子观点未匹配: %s",
                                   title[:30], len(unmatched_keys), unmatched_keys)

            claims.append(ClaimResult(
                claim_title=title,
                evidences=evidences,
            ))

        # 过滤掉没有证据的空论点
        empty_claims = [c.claim_title for c in claims if not c.evidences]
        if empty_claims:
            logger.warning("过滤 %d 个空论点: %s", len(empty_claims),
                           [t[:30] for t in empty_claims])
        claims = [c for c in claims if c.evidences]

        # 每个论点内的证据按评分降序排列
        for cl in claims:
            cl.evidences.sort(
                key=lambda e: e.score.total_score if e.score else 0,
                reverse=True,
            )

        # 论点之间按最高证据分降序排列（最强论点排最前）
        claims.sort(
            key=lambda c: max((e.score.total_score for e in c.evidences if e.score), default=0),
            reverse=True,
        )

        output = ResearchOutput(
            query=self.query,
            claims=claims,
            total_evidences=sum(len(c.evidences) for c in claims),
            search_rounds=self._search_count,
            completed_at=datetime.now(),
        )

        # 递增任务计数（P5-J1 周期整合用）
        try:
            from memory.source_memory import get_source_repository
            task_count = get_source_repository().increment_task_count()
            logger.info("Main Agent 提交结果: %d 论点, %d 证据 (累计任务 #%d)",
                         len(claims), output.total_evidences, task_count)
        except Exception as e:
            logger.warning("任务计数更新失败: %s", e)
            logger.info("Main Agent 提交结果: %d 论点, %d 证据",
                         len(claims), output.total_evidences)

        return StepOutcome(data=output, should_exit=True)

    # ── 内部辅助 ──────────────────────────────────────────

    def _record_sources(
        self,
        evidence_items: list[EvidenceItem],
        accepted_sources: list[str],
        rejected_sources: list[str],
        topic_context: str,
    ) -> None:
        """将本次搜索涉及的来源写入长期记忆（P4-H6 记忆候选选择）。

        三类决策：
        1. 值得长期记忆（score >= memory_threshold）：写入或更新记忆
        2. 本次局部可用（quality_threshold <= score < memory_threshold）：
           - 已在记忆中 → 更新统计（为升降级提供数据）
           - 不在记忆中 → 跳过，不值得记住
        3. 未通过质量阈值（rejected）：仅更新已在记忆中的来源（为降级提供数据）

        同一域名取最高分记录。
        """
        from memory.source_memory import record_source, get_source_repository
        from urllib.parse import urlparse

        memory_threshold = cfg.agent.memory_threshold
        topics = [topic_context] if topic_context else []
        repo = get_source_repository()

        # 从证据中收集 accepted 来源的域名、最高评分和领域标签
        domain_scores: dict[str, float] = {}
        domain_tags: dict[str, set[str]] = {}  # domain → 合并的 tags
        for ev in evidence_items:
            if ev.source_domain and ev.score:
                prev = domain_scores.get(ev.source_domain, 0.0)
                domain_scores[ev.source_domain] = max(prev, ev.score.total_score)
                # 收集该域名关联的所有 domain_tags
                if ev.score.domain_tags:
                    if ev.source_domain not in domain_tags:
                        domain_tags[ev.source_domain] = set()
                    domain_tags[ev.source_domain].update(ev.score.domain_tags)

        # 处理 accepted 来源
        recorded_domains: set[str] = set()
        for url in accepted_sources:
            try:
                domain = urlparse(url).netloc
            except Exception:
                continue
            if not domain or domain in recorded_domains:
                continue
            recorded_domains.add(domain)

            score = domain_scores.get(domain, 70.0)
            already_in_memory = repo.find_by_domain(domain) is not None

            if score >= memory_threshold or already_in_memory:
                try:
                    tags = list(domain_tags.get(domain, set()))
                    record_source(domain=domain, score=score, passed=True,
                                  topics=topics, domain_tags=tags, repo=repo)
                except Exception as e:
                    logger.warning("记录来源失败 (%s): %s", domain, e)
            else:
                logger.debug("来源 %s 评分 %.0f 低于记忆阈值 %d，仅本次可用",
                             domain, score, memory_threshold)

        # 处理 rejected 来源：仅更新已在记忆中的来源
        rejected_domains: set[str] = set()
        for url in rejected_sources:
            try:
                domain = urlparse(url).netloc
            except Exception:
                continue
            if not domain or domain in rejected_domains:
                continue
            rejected_domains.add(domain)

            if repo.find_by_domain(domain) is None:
                continue

            cached = self._page_cache.get(url)
            score = cached[1].total_score if cached and cached[1] else 40.0
            tags = list(cached[1].domain_tags) if cached and cached[1] and cached[1].domain_tags else []
            try:
                record_source(domain=domain, score=score, passed=False,
                              topics=topics, domain_tags=tags, repo=repo)
            except Exception as e:
                logger.warning("记录来源失败 (%s): %s", domain, e)

    def _build_fallback_output(self) -> ResearchOutput:
        """兜底：把所有证据按 subclaim 分组，并按评分排序。"""
        claims = []
        for subclaim, indices in self._subclaim_evidence_map.items():
            evidences = [self._all_evidence[i] for i in indices if i < len(self._all_evidence)]
            if evidences:
                # 论点内证据按评分降序
                evidences.sort(
                    key=lambda e: e.score.total_score if e.score else 0,
                    reverse=True,
                )
                claims.append(ClaimResult(
                    claim_title=subclaim[:50],
                    evidences=evidences,
                ))

        # 论点之间按最高证据分降序
        claims.sort(
            key=lambda c: max((e.score.total_score for e in c.evidences if e.score), default=0),
            reverse=True,
        )

        return ResearchOutput(
            query=self.query,
            claims=claims,
            total_evidences=sum(len(c.evidences) for c in claims),
            search_rounds=self._search_count,
            completed_at=datetime.now(),
        )

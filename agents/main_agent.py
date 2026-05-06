"""
Deep Truth Search — Main Agent

全局研究控制 Agent。通过 Agent Loop 自主拆解子观点、
调用 Search Agent、判断覆盖度、组织最终结果。

用法（API 层调用）：
    from agents.main_agent import run_research

    output = await run_research("AI在2024年取得重大突破", on_status=callback)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional, Union

from config import cfg
from models import ResearchOutput

logger = logging.getLogger(__name__)

# 状态回调类型：(task_id, msg) -> None，msg 可以是 str 或 dict（结构化事件）
StatusCallback = Optional[Callable[[str, Union[str, dict]], None]]


async def run_research(
    query: str,
    *,
    on_status: StatusCallback = None,
) -> ResearchOutput:
    """Main Agent 核心入口：Agent Loop 版本。

    LLM 自主拆解子观点、决定搜索方向、判断覆盖度、组织最终结果。

    Args:
        query: 用户输入的观点
        on_status: 可选的状态回调 (task_id, status_text)

    Returns:
        ResearchOutput 结构化结果
    """
    from agents.agent_loop import run_agent_loop
    from agents.main_handler import MainAgentHandler

    logger.info("Main Agent 开始研究: %s", query[:80])

    # 包装 on_status 以兼容 (task_id, msg) 签名，msg 可以是 str 或 dict
    def _status_callback(msg: str | dict) -> None:
        if on_status:
            on_status("", msg)

    handler = MainAgentHandler(query=query, on_status=_status_callback)

    if on_status:
        on_status("", "正在分析观点，规划搜索方向...")

    result = await run_agent_loop(
        handler=handler,
        user_message=f"请研究以下观点，搜索证据：\n{query}",
        max_turns=cfg.agent.main_max_turns,
        on_status=_status_callback,
    )

    if isinstance(result, ResearchOutput):
        logger.info("Main Agent 完成: %d 论点, %d 证据",
                     len(result.claims), result.total_evidences)
        return result

    # 兜底：如果 loop 结束但没有通过 submit_results 返回
    logger.warning("Main Agent Loop 未正常结束，构建兜底结果")
    fallback = handler._build_fallback_output()
    return fallback

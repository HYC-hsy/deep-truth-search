"""
Deep Truth Search — 通用 Agent Loop 引擎

参考 pc-agent-loop 设计模式：
  while turn < max_turns:
    response = LLM(messages, tools)   ← think + act
    outcome  = handler.dispatch(tc)   ← execute
    messages.append(result)           ← observe
    if should_exit: return

Main Agent 和 Search Agent 各自作为 handler 插入此引擎。

用法：
    result = await run_agent_loop(handler, "user message", max_turns=10)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── 核心数据类型 ──────────────────────────────────────────────


@dataclass
class StepOutcome:
    """单次工具执行的结果。

    Attributes:
        data: 回馈给 LLM 的观察数据（会 JSON 序列化后作为 tool result）
        should_exit: True 时立即终止循环，data 作为最终返回值
    """
    data: Any = None
    should_exit: bool = False


# ── Handler 接口 ──────────────────────────────────────────────


@runtime_checkable
class AgentHandler(Protocol):
    """Agent Loop Handler 接口（duck-typed）。

    Main Agent 和 Search Agent 各实现一个 handler。
    """

    def get_system_prompt(self) -> str:
        """返回 system prompt。"""
        ...

    def get_tools_schema(self) -> list[dict]:
        """返回 OpenAI function calling 格式的工具 schema。"""
        ...

    async def dispatch(self, tool_name: str, args: dict) -> StepOutcome:
        """分发工具调用到具体实现。"""
        ...

    def on_turn_start(self, turn: int, max_turns: int) -> Optional[str]:
        """每轮开始前调用，可返回注入消息（软升级警告等）。返回 None 则不注入。"""
        ...

    def on_max_turns(self) -> Any:
        """超过 max_turns 时调用，返回兜底结果。"""
        ...


StatusCallback = Optional[Callable[[str], None]]
StepCallback = Optional[Callable[[str, str], None]]  # (event_type, content)


# ── Agent Loop 主函数 ────────────────────────────────────────


async def run_agent_loop(
    handler: AgentHandler,
    user_message: str,
    *,
    max_turns: int = 10,
    on_status: StatusCallback = None,
    on_step: StepCallback = None,
) -> Any:
    """通用异步 Agent Loop。

    每轮：
    1. 调用 LLM（带 tools）→ LLM 思考并选择工具
    2. 如果 LLM 无工具调用（纯文本响应）→ 循环结束
    3. 逐个 dispatch 工具调用 → 收集观察结果
    4. 将 assistant 消息 + tool results 追加到 messages
    5. 检查 should_exit

    Args:
        handler: 实现 AgentHandler 接口的处理器
        user_message: 初始用户消息
        max_turns: 最大循环轮次
        on_status: 状态回调（可选）
        on_step: 步骤级回调（可选），用于实时推送搜索过程日志

    Returns:
        handler 通过 StepOutcome(should_exit=True) 返回的最终数据，
        或超时兜底结果
    """
    from llm.llm_client import call_llm_with_tools

    system_prompt = handler.get_system_prompt()
    tools_schema = handler.get_tools_schema()

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(1, max_turns + 1):
        # 软升级警告注入
        injection = handler.on_turn_start(turn, max_turns)
        if injection:
            messages.append({"role": "user", "content": injection})

        if on_status:
            on_status(f"Agent 第 {turn}/{max_turns} 轮")

        logger.info("Agent Loop turn %d/%d", turn, max_turns)

        # ── Think + Act: 调用 LLM ──
        try:
            response = await call_llm_with_tools(
                messages=messages,
                tools=tools_schema,
            )
        except Exception as e:
            logger.error("Agent Loop LLM 调用失败 (turn %d): %s", turn, e)
            # 跳过本轮，下轮重试
            messages.append({"role": "user", "content": f"上一轮 LLM 调用失败: {e}。请继续。"})
            continue

        # 构建 assistant 消息
        assistant_msg: dict = {"role": "assistant"}
        if response.content:
            assistant_msg["content"] = response.content
            logger.info("Turn %d think: %s", turn, response.content[:200])
            if on_step:
                on_step("think", response.content)
        else:
            logger.debug("Turn %d: content 为空，无 think 事件", turn)
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # 无工具调用 → LLM 纯文本响应
        if not response.tool_calls:
            logger.info("Turn %d: LLM 无工具调用，调用兜底", turn)
            return handler.on_max_turns()

        # ── Execute + Observe ──
        # 同一 turn 内多个 visit_page 并行执行，其余串行
        final_result = None
        tool_calls = response.tool_calls

        # 将连续的 visit_page 分组为可并行批次
        batches = _group_tool_calls(tool_calls)

        for batch in batches:
            if final_result is not None:
                break

            if len(batch) == 1:
                # 单个工具调用，串行执行
                tc = batch[0]
                logger.info("Turn %d: 调用工具 %s(%s)",
                            turn, tc.function_name,
                            json.dumps(tc.arguments, ensure_ascii=False)[:100])
                if on_step:
                    on_step("tool_call", json.dumps(
                        {"tool": tc.function_name, "args": tc.arguments},
                        ensure_ascii=False))
                try:
                    outcome = await handler.dispatch(tc.function_name, tc.arguments)
                except Exception as e:
                    logger.error("工具 %s 执行异常: %s", tc.function_name, e)
                    outcome = StepOutcome(data={"error": str(e)})

                if on_step:
                    on_step("tool_result", json.dumps(
                        {"tool": tc.function_name,
                         "data": outcome.data if isinstance(outcome.data, (dict, list)) else str(outcome.data)},
                        ensure_ascii=False, default=str))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _serialize_outcome(outcome),
                })

                if outcome.should_exit:
                    logger.info("Turn %d: 工具 %s 触发退出", turn, tc.function_name)
                    final_result = outcome.data
            else:
                # 多个 visit_page 并行执行
                logger.info("Turn %d: 并行执行 %d 个 visit_page", turn, len(batch))
                for tc in batch:
                    logger.info("  → %s(%s)", tc.function_name,
                                json.dumps(tc.arguments, ensure_ascii=False)[:100])
                    if on_step:
                        on_step("tool_call", json.dumps(
                            {"tool": tc.function_name, "args": tc.arguments},
                            ensure_ascii=False))

                async def _dispatch_one(tc_item):
                    try:
                        return await handler.dispatch(tc_item.function_name, tc_item.arguments)
                    except Exception as e:
                        logger.error("工具 %s 执行异常: %s", tc_item.function_name, e)
                        return StepOutcome(data={"error": str(e)})

                outcomes = await asyncio.gather(*[_dispatch_one(tc) for tc in batch])

                # 按原始顺序追加结果
                for tc, outcome in zip(batch, outcomes):
                    if on_step:
                        on_step("tool_result", json.dumps(
                            {"tool": tc.function_name,
                             "data": outcome.data if isinstance(outcome.data, (dict, list)) else str(outcome.data)},
                            ensure_ascii=False, default=str))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _serialize_outcome(outcome),
                    })
                    if outcome.should_exit:
                        logger.info("Turn %d: 工具 %s 触发退出", turn, tc.function_name)
                        final_result = outcome.data
                        break

        if final_result is not None:
            return final_result

    # 超过 max_turns
    logger.warning("Agent Loop 达到最大轮次 %d，调用兜底", max_turns)
    return handler.on_max_turns()


# ── 并行分组辅助 ──────────────────────────────────────────────

# 可以安全并行的工具名（无互相依赖、无 should_exit）
_PARALLELIZABLE_TOOLS = {"visit_page", "web_search"}


def _group_tool_calls(tool_calls: list) -> list[list]:
    """将同一 turn 的 tool calls 分组。

    连续的可并行工具（如多个 visit_page）合为一个批次，
    其他工具各自独立为一个批次，保持原始顺序。

    示例: [visit, visit, search, visit, visit, submit]
         → [[visit, visit], [search], [visit, visit], [submit]]
    """
    batches: list[list] = []
    current_parallel: list = []

    for tc in tool_calls:
        if tc.function_name in _PARALLELIZABLE_TOOLS:
            current_parallel.append(tc)
        else:
            if current_parallel:
                batches.append(current_parallel)
                current_parallel = []
            batches.append([tc])

    if current_parallel:
        batches.append(current_parallel)

    return batches


def _serialize_outcome(outcome: StepOutcome) -> str:
    """将 StepOutcome.data 序列化为字符串。"""
    if isinstance(outcome.data, (dict, list)):
        return json.dumps(outcome.data, ensure_ascii=False, default=str)
    return str(outcome.data) if outcome.data is not None else ""

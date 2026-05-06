"""
Deep Truth Search — 异步任务执行与状态管理

Main Agent loop 耗时数分钟，需要后台异步执行。
此模块管理任务的提交、状态跟踪和结果获取。
支持 SSE 事件推送，前端可实时接收搜索进度。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from models import ResearchOutput, TaskStatus

logger = logging.getLogger(__name__)


class TaskInfo:
    """单个任务的状态信息"""

    def __init__(self, task_id: str, query: str, session_id: str = ""):
        self.task_id = task_id
        self.query = query
        self.session_id = session_id  # 关联的会话 ID
        self.status: TaskStatus = TaskStatus.PENDING
        self.status_message: str = ""
        self.result: ResearchOutput | None = None
        self.error: str | None = None
        self.created_at: datetime = datetime.now()
        self.completed_at: datetime | None = None
        self.last_active_at: datetime = datetime.now()  # 后端最后活跃时间
        self._sse_queues: list[asyncio.Queue] = []  # SSE 订阅者队列

    def touch(self, message: str | dict = "") -> None:
        """更新活跃时间，可选更新状态消息。支持结构化事件（dict）。"""
        self.last_active_at = datetime.now()
        if isinstance(message, dict):
            # 结构化事件（batch_start / search_start / search_done）直接广播
            self.status_message = message.get("subclaim", "") or str(message.get("type", ""))
            self._broadcast(message)
        elif message:
            self.status_message = message
            self._broadcast({"type": "status", "message": message})

    def _broadcast(self, event: dict) -> None:
        """向所有 SSE 订阅者推送事件。"""
        for q in self._sse_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # 队列满时丢弃，避免阻塞

    def subscribe(self) -> asyncio.Queue:
        """创建并返回一个 SSE 事件队列。"""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """移除 SSE 订阅。"""
        try:
            self._sse_queues.remove(q)
        except ValueError:
            pass


class TaskManager:
    """内存中的任务状态管理器"""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}

    def submit(self, query: str, session_id: str = "") -> str:
        """提交新的研究任务，返回 task_id。"""
        task_id = uuid.uuid4().hex[:12]
        info = TaskInfo(task_id=task_id, query=query, session_id=session_id)
        self._tasks[task_id] = info

        # 启动后台执行
        asyncio.create_task(self._run(info))

        logger.info("任务已提交: %s (query=%s)", task_id, query[:60])
        return task_id

    def get(self, task_id: str) -> TaskInfo | None:
        return self._tasks.get(task_id)

    async def _run(self, info: TaskInfo) -> None:
        """后台执行 Main Agent loop。"""
        from agents.main_agent import run_research

        def on_status(task_id: str, msg: str | dict) -> None:
            info.touch(msg)

        try:
            info.status = TaskStatus.SEARCHING
            info.touch("正在分析观点，规划搜索方向...")

            result = await run_research(
                info.query,
                on_status=on_status,
            )

            info.result = result
            info.status = TaskStatus.DONE
            info.status_message = "搜索完成"
            info.completed_at = datetime.now()
            info._broadcast({
                "type": "done",
                "message": "搜索完成",
                "result": result.model_dump(mode="json"),
            })
            logger.info("任务完成: %s, %d 条证据", info.task_id, result.total_evidences)

            # 保存 agent 消息到会话
            if info.session_id:
                try:
                    import json as _json
                    from api.session_manager import get_session_manager
                    from models import SessionMessage
                    sm = get_session_manager()
                    sm.add_message(info.session_id, SessionMessage(
                        role="agent",
                        content=_json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                        task_id=info.task_id,
                    ))
                except Exception:
                    logger.exception("保存 agent 消息到会话失败")

        except Exception as exc:
            logger.exception("任务执行失败: %s", info.task_id)
            info.status = TaskStatus.ERROR
            info.error = str(exc)
            info.status_message = f"搜索失败: {exc}"
            info.completed_at = datetime.now()
            info._broadcast({
                "type": "error",
                "message": str(exc),
            })

    def to_status_dict(self, info: TaskInfo) -> dict:
        """将任务信息序列化为 API 响应格式。"""
        data: dict[str, Any] = {
            "task_id": info.task_id,
            "status": info.status.value,
            "status_message": info.status_message,
            "query": info.query,
            "last_active_at": info.last_active_at.isoformat(),
        }

        if info.status == TaskStatus.DONE and info.result:
            data["result"] = info.result.model_dump(mode="json")

        if info.status == TaskStatus.ERROR and info.error:
            data["error"] = info.error

        return data


# 模块级单例
task_manager = TaskManager()

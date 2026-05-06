"""
Deep Truth Search — 启动入口

运行方式：python main.py
启动后自动打开浏览器，访问 Web UI。
"""

import asyncio
import json
import logging
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import cfg

# ── 日志配置 ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deep_truth_search")

# ── FastAPI 应用 ──────────────────────────────────────────────

app = FastAPI(
    title="Deep Truth Search",
    description="自进化证据查找 Deep Research 系统",
    version="0.1.0",
)

# ── 静态文件 ──────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(cfg.root / "ui")), name="static")


# ── 页面路由 ──────────────────────────────────────────────────

@app.get("/")
async def index():
    """主页面"""
    return FileResponse(str(cfg.root / "ui" / "index.html"))


# ── API 路由（MVP-D1/D2）──────────────────────────────────────

from api.task_manager import task_manager
from api.session_manager import get_session_manager
from models import SessionMessage


@app.post("/api/research")
async def create_research(body: dict):
    """提交研究任务，返回 task_id。Main Agent 在后台异步执行。"""
    query = body.get("query", "")
    if not query or not query.strip():
        return {"error": "请输入观点内容", "status": "error"}

    session_id = body.get("session_id", "")

    # 向会话追加 user 消息
    if session_id:
        sm = get_session_manager()
        sm.add_message(session_id, SessionMessage(role="user", content=query.strip()))

    task_id = task_manager.submit(query.strip(), session_id=session_id)
    return {"task_id": task_id, "status": "pending", "query": query}


# ── 会话 API（P5-K1/K2）──────────────────────────────────────


@app.post("/api/sessions")
async def create_session():
    """创建新会话。"""
    sm = get_session_manager()
    session = sm.create()
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
    }


@app.get("/api/sessions")
async def list_sessions():
    """列出所有会话摘要。"""
    sm = get_session_manager()
    return sm.list_all()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """加载完整会话。"""
    sm = get_session_manager()
    session = sm.get(session_id)
    if session is None:
        return {"error": "会话不存在", "status": "error"}
    return session.model_dump(mode="json")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话。"""
    sm = get_session_manager()
    ok = sm.delete(session_id)
    if not ok:
        return {"error": "会话不存在", "status": "error"}
    return {"status": "ok"}


@app.get("/api/research/{task_id}/status")
async def get_research_status(task_id: str):
    """查询任务状态和结果。前端轮询此接口（SSE 的备用方案）。"""
    info = task_manager.get(task_id)
    if info is None:
        return {"error": "任务不存在", "status": "error", "task_id": task_id}
    return task_manager.to_status_dict(info)


@app.get("/api/research/{task_id}/stream")
async def stream_research_status(task_id: str):
    """SSE 端点：实时推送搜索进度事件。

    事件类型：
    - status: 搜索状态更新（正在拆解观点、正在搜索子观点X...）
    - done:   搜索完成，附带完整结果
    - error:  搜索出错
    """
    info = task_manager.get(task_id)
    if info is None:
        return {"error": "任务不存在", "status": "error", "task_id": task_id}

    logger.info("SSE 连接建立: task=%s, status=%s", task_id, info.status.value)

    async def event_generator():
        # 如果任务已完成，直接发送结果并关闭
        if info.status.value == "done" and info.result:
            yield _sse_format("done", {
                "message": "搜索完成",
                "result": info.result.model_dump(mode="json"),
            })
            return
        if info.status.value == "error":
            yield _sse_format("error", {"message": info.error or "未知错误"})
            return

        # 发送当前状态作为初始事件
        if info.status_message:
            yield _sse_format("status", {"message": info.status_message})

        # 订阅后续事件
        queue = info.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield ": heartbeat\n\n"
                    continue

                event_type = event.get("type", "status")
                # 结构化进度事件（batch_start 等）统一用 SSE event: status 发送
                sse_event = event_type if event_type in ("done", "error") else "status"
                yield _sse_format(sse_event, event)

                # 终止事件：done 或 error
                if event_type in ("done", "error"):
                    break
        finally:
            info.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_format(event_type: str, data: dict) -> str:
    """将事件格式化为 SSE 协议文本。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── 自动打开浏览器 ────────────────────────────────────────────

def _open_browser():
    """延迟打开浏览器，等服务启动完成"""
    import time
    time.sleep(1.5)
    url = f"http://{cfg.web.host}:{cfg.web.port}"
    logger.info("正在打开浏览器: %s", url)
    webbrowser.open(url)


# ── 启动环境校验 ──────────────────────────────────────────────

def _check_env():
    """前置环境检查，缺失关键配置时输出中文错误提示并退出。"""
    errors: list[str] = []

    # .env 文件是否存在
    env_path = cfg.root / ".env"
    if not env_path.exists():
        errors.append(
            "[配置缺失] 未找到 .env 文件。\n"
            "  请先复制模板：cp .env.example .env\n"
            "  然后填入你的 API key。"
        )
        _print_errors_and_exit(errors)

    # LLM API key
    if not cfg.llm.api_key:
        errors.append(
            "[配置缺失] LLM_API_KEY 未填写。\n"
            "  请在 .env 文件中设置 LLM_API_KEY=你的key"
        )

    # 搜索 API key（警告而非退出，MVP 阶段允许先不配）
    if not cfg.search.api_key or cfg.search.api_key.startswith(("tvly-", "your-")):
        logger.warning(
            "SEARCH_API_KEY 未配置或仍为模板值。搜索功能将不可用，请在 .env 中填写。"
        )

    if errors:
        _print_errors_and_exit(errors)

    logger.info("环境校验通过")
    logger.info("主 LLM: %s (%s)", cfg.llm.model, cfg.llm.base_url)
    if cfg.scoring_llm.is_configured():
        logger.info("评分 LLM: %s (%s)", cfg.scoring_llm.model, cfg.scoring_llm.base_url)
    else:
        logger.info("评分 LLM: 使用主 LLM")


def _print_errors_and_exit(errors: list[str]):
    """输出错误信息并退出"""
    print("\n" + "=" * 60)
    print("  Deep Truth Search — 启动失败")
    print("=" * 60)
    for err in errors:
        print(f"\n{err}")
    print("\n" + "=" * 60 + "\n")
    sys.exit(1)


# ── 启动 ──────────────────────────────────────────────────────

def main():
    _check_env()

    host = cfg.web.host
    port = cfg.web.port

    logger.info("Deep Truth Search 启动中...")
    logger.info("地址: http://%s:%d", host, port)

    if cfg.web.auto_open:
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

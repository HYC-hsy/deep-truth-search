"""
Deep Truth Search — 会话管理（P5-K1）

每个会话持久化为一个 JSON 文件: data/sessions/{session_id}.json
支持创建、读取、追加消息、列出摘要、删除。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

# 会话保留天数（超过此时间的会话自动清理）
SESSION_RETENTION_DAYS = 90

from models import Session, SessionMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """会话 CRUD，JSON 文件持久化（每会话一个文件）"""

    def __init__(self, storage_dir: Path) -> None:
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ── 公开方法 ──────────────────────────────────────────────

    def create(self) -> Session:
        """创建新会话，返回空会话对象。"""
        session = Session(
            session_id=uuid.uuid4().hex[:12],
            title="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
        )
        self._save(session)
        logger.info("会话已创建: %s", session.session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        """按 ID 加载完整会话。"""
        return self._load(session_id)

    def list_all(self) -> list[dict]:
        """列出所有会话摘要（id, title, created_at, updated_at, message_count）。

        按 updated_at 降序排列。自动清理超过 SESSION_RETENTION_DAYS 的过期会话。
        """
        summaries: list[dict] = []
        cutoff = datetime.now() - timedelta(days=SESSION_RETENTION_DAYS)

        for fname in self._dir.iterdir():
            if not fname.suffix == ".json" or fname.name.startswith("_"):
                continue
            try:
                raw = json.loads(fname.read_text(encoding="utf-8"))
                updated = raw.get("updated_at", "")

                # 自动清理过期会话
                if updated and updated < cutoff.isoformat():
                    fname.unlink()
                    logger.info("自动清理过期会话: %s", fname.name)
                    continue

                summaries.append({
                    "session_id": raw["session_id"],
                    "title": raw.get("title", ""),
                    "created_at": raw.get("created_at", ""),
                    "updated_at": updated,
                    "message_count": len(raw.get("messages", [])),
                })
            except Exception:
                logger.warning("跳过损坏的会话文件: %s", fname.name)
        summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return summaries

    def add_message(self, session_id: str, msg: SessionMessage) -> Session | None:
        """向会话追加一条消息。自动生成标题（首条 user 消息前 40 字符）。"""
        with self._lock:
            session = self._load(session_id)
            if session is None:
                logger.warning("会话不存在: %s", session_id)
                return None

            session.messages.append(msg)
            session.updated_at = datetime.now()

            # 自动标题：首条 user 消息
            if not session.title and msg.role == "user":
                session.title = msg.content[:40].strip()

            self._save(session)
            return session

    def delete(self, session_id: str) -> bool:
        """删除会话文件。"""
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            logger.info("会话已删除: %s", session_id)
            return True
        return False

    # ── 内部方法 ──────────────────────────────────────────────

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return Session.model_validate(raw)
        except Exception:
            logger.exception("加载会话失败: %s", session_id)
            return None

    def _save(self, session: Session) -> None:
        """原子写入：写临时文件再 rename。"""
        path = self._path(session.session_id)
        tmp = path.with_suffix(".tmp")
        data = session.model_dump(mode="json")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # Windows 上 rename 目标已存在时需先删除
        if path.exists():
            path.unlink()
        tmp.rename(path)


# ── 模块级单例 ──────────────────────────────────────────────────

_default_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """获取默认的会话管理器实例（懒加载单例）。"""
    global _default_manager
    if _default_manager is None:
        from config import cfg
        storage_dir = cfg.root / "data" / "sessions"
        _default_manager = SessionManager(storage_dir)
    return _default_manager

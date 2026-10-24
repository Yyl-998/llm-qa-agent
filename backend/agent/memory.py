"""会话记忆管理：多轮对话上下文按会话隔离，滑动窗口控制上下文长度。"""
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import BaseMessage

from backend.config import MAX_HISTORY_TURNS


@dataclass
class Session:
    id: str
    title: str
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[BaseMessage] = field(default_factory=list)


class SessionStore:
    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_turns = max_turns

    def create(self) -> Session:
        with self._lock:
            s = Session(id=uuid.uuid4().hex[:8], title="新会话")
            self._sessions[s.id] = s
            return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            s = self._sessions.get(session_id)
            if s:
                return s
        return self.create()

    def list_sessions(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def append(self, session_id: str, messages: list[BaseMessage]):
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.messages.extend(messages)

    def windowed_history(self, session: Session) -> list[BaseMessage]:
        """最近 N 轮（2N 条消息）作为 Agent 上下文。"""
        return list(session.messages[-self.max_turns * 2:])

    def set_title_if_new(self, session_id: str, first_message: str):
        s = self._sessions.get(session_id)
        if s and s.title == "新会话" and first_message:
            s.title = first_message[:20]


store = SessionStore()

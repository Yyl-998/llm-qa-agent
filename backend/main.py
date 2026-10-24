"""FastAPI 后端服务：SSE 流式对话、会话管理、知识库文档管理。"""
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.agent import core
from backend.agent.memory import store
from backend.config import ALLOWED_EXTS, DOCS_DIR
from backend.rag.knowledge import knowledge_base
from backend.utils.loader import load_document, split_documents

app = FastAPI(title="LLM 智能问答 Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 会话 ----------------

@app.post("/api/sessions")
def create_session():
    s = store.create()
    return {"id": s.id, "title": s.title, "created_at": _fmt(s.created_at)}


@app.get("/api/sessions")
def list_sessions():
    return [
        {"id": s.id, "title": s.title, "created_at": _fmt(s.created_at)}
        for s in store.list_sessions()
    ]


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str):
    s = store.get(session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    return {
        "session_id": s.id,
        "title": s.title,
        "messages": [
            {"role": "human" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in s.messages
        ],
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not store.delete(session_id):
        raise HTTPException(404, "会话不存在")
    return {"ok": True}


# ---------------- 对话（SSE 流式） ----------------

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = store.get_or_create(req.session_id)
    store.set_title_if_new(session.id, req.message)
    return StreamingResponse(
        core.stream_chat(session, req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- 知识库文档 ----------------

@app.get("/api/documents")
def list_documents():
    return [{"name": n} for n in core.list_documents()]


@app.post("/api/documents")
async def upload_document(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的文件类型 {ext}，仅支持 pdf / txt / md / docx")
    path = DOCS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    data = await file.read()
    for attempt in range(3):
        try:
            path.write_bytes(data)
            break
        except PermissionError:
            if attempt == 2:
                raise HTTPException(500, "文件写入被占用（可能被杀毒软件暂时锁定），请稍后重试")
            time.sleep(0.4)
    try:
        chunks = split_documents(load_document(path))
        if not chunks:
            raise HTTPException(400, "文档内容为空或无法解析")
        n = knowledge_base.add_document(chunks)
    except Exception as e:
        path.unlink(missing_ok=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(500, f"文档入库失败：{e}")
    return {"ok": True, "name": file.filename, "chunks": n}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    if "/" in name or "\\" in name:
        raise HTTPException(400, "非法文件名")
    path = DOCS_DIR / name
    if not path.exists():
        raise HTTPException(404, "文档不存在")
    path.unlink()
    knowledge_base.remove_document(name)
    return {"ok": True}


# ---------------- 健康检查 ----------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents": len(core.list_documents()),
        "vector_index_ready": not knowledge_base.is_empty(),
    }

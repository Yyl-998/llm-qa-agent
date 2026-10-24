"""Streamlit 前端：智能问答 Agent 交互界面。

功能：多会话管理、SSE 流式渲染、工具调用过程展示、知识库文档上传与管理。
启动：streamlit run frontend/app.py
"""
import json
import os
import re

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

TOOL_LABELS = {
    "get_weather": "天气查询",
    "web_search": "网页搜索",
    "knowledge_search": "知识库检索",
    "summarize_document": "文档摘要",
}

st.set_page_config(page_title="LLM 智能问答 Agent", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions" not in st.session_state:
    st.session_state.sessions = []


def api(path: str, method: str = "get", timeout: int = 300, **kwargs):
    try:
        r = getattr(requests, method)(f"{API_BASE}{path}", timeout=timeout, **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"接口错误 {r.status_code}：{detail}")
            return None
        return r
    except requests.RequestException as e:
        st.error(f"无法连接后端服务（{API_BASE}），请先启动 FastAPI：{e}")
        return None


def refresh_sessions():
    r = api("/api/sessions")
    st.session_state.sessions = r.json() if r else []


# ---------------- 侧边栏：会话 + 知识库 ----------------

with st.sidebar:
    st.title("⚙️ 控制台")

    st.subheader("💬 会话")
    if st.button("➕ 新建会话", use_container_width=True):
        r = api("/api/sessions", method="post")
        if r:
            st.session_state.session_id = r.json()["id"]
            st.session_state.messages = []
            refresh_sessions()
            st.rerun()

    for s in st.session_state.sessions:
        col1, col2 = st.columns([4, 1])
        active = s["id"] == st.session_state.session_id
        if col1.button(
            ("🟢 " if active else "") + s["title"],
            key=f"s_{s['id']}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            if not active:
                st.session_state.session_id = s["id"]
                r = api(f"/api/sessions/{s['id']}/history")
                st.session_state.messages = r.json()["messages"] if r else []
                st.rerun()
        if col2.button("🗑", key=f"d_{s['id']}"):
            api(f"/api/sessions/{s['id']}", method="delete")
            if active:
                st.session_state.session_id = None
                st.session_state.messages = []
            refresh_sessions()
            st.rerun()

    st.divider()
    st.subheader("📚 知识库（RAG）")
    uploaded = st.file_uploader(
        "上传文档（PDF / TXT / MD / DOCX）",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("上传并入库", type="primary", use_container_width=True):
        for f in uploaded:
            r = api("/api/documents", method="post", files={"file": (f.name, f.getvalue())})
            if r:
                st.toast(f"《{f.name}》入库成功：切分出 {r.json()['chunks']} 个片段")
        st.rerun()

    r = api("/api/documents")
    if r:
        for d in r.json():
            c1, c2 = st.columns([4, 1])
            c1.caption("📄 " + re.sub(r"^[0-9a-f]{8}_", "", d["name"]))
            if c2.button("✖", key=f"doc_{d['name']}"):
                api(f"/api/documents/{d['name']}", method="delete")
                st.rerun()

# ---------------- 主区域：对话 ----------------

st.title("🤖 LLM 智能问答 Agent")
st.caption("LangChain 多轮对话 · Function Calling 工具调度 · RAG 知识检索 · SSE 流式输出")

for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "human" else "assistant"):
        st.markdown(m["content"])

user_input = st.chat_input("请输入你的问题，例如：北京今天天气怎么样？")
if user_input:
    if not st.session_state.session_id:
        r = api("/api/sessions", method="post")
        if not r:
            st.stop()
        st.session_state.session_id = r.json()["id"]
        refresh_sessions()

    st.session_state.messages.append({"role": "human", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    answer = ""
    error = False
    with st.chat_message("assistant"):
        status_box = st.empty()
        msg_box = st.empty()
        tool_lines: list[str] = []
        try:
            with requests.post(
                f"{API_BASE}/api/chat",
                json={"session_id": st.session_state.session_id, "message": user_input},
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    evt = json.loads(line[len("data: "):])
                    t = evt.get("type")
                    if t == "token":
                        answer += evt["content"]
                        msg_box.markdown(answer + " ▌")
                    elif t == "tool_start":
                        name = TOOL_LABELS.get(evt["name"], evt["name"])
                        tool_lines.append(f"🔧 调用工具 **{name}** · 参数：`{evt.get('input', '')}`")
                        status_box.info("  \n".join(tool_lines) + "  \n\n⏳ 执行中…")
                    elif t == "tool_end":
                        name = TOOL_LABELS.get(evt["name"], evt["name"])
                        out = evt.get("output", "").replace("\n", " ")[:120]
                        tool_lines.append(f"&nbsp;&nbsp;&nbsp;↳ ✅ {name} 完成：{out}…")
                        status_box.info("  \n".join(tool_lines))
                    elif t == "error":
                        st.error(f"Agent 执行出错：{evt.get('content')}")
                        error = True
                    elif t == "done":
                        if not answer:
                            answer = evt.get("content", "")
        except requests.RequestException as e:
            st.error(f"请求失败：{e}")
            error = True

    if answer or not error:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer or "（模型未返回内容）"}
        )
    st.rerun()

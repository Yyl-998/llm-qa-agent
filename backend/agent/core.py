"""Agent 核心：基于 LangChain Function Calling 的多轮对话 Agent。"""
import json
from datetime import date

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.agent.memory import Session, store
from backend.agent.tools.knowledge import knowledge_search, list_documents, summarize_document
from backend.agent.tools.search import web_search
from backend.agent.tools.weather import get_weather
from backend.llm import get_llm

TOOLS = [get_weather, web_search, knowledge_search, summarize_document]

SYSTEM_TEMPLATE = """你是一个智能问答助手「小智」。当前日期：{date}。

你可以使用以下工具：
1. get_weather —— 查询城市实时天气与未来 3 天预报；
2. web_search —— 搜索互联网，获取实时资讯或你不确定的信息；
3. knowledge_search —— 在用户上传的知识库文档中做语义检索（RAG）；
4. summarize_document —— 对知识库中的文档进行摘要总结。

要求：
- 根据用户意图自主决定是否调用工具、调用哪个工具；需要多步时逐步调用；
- 用户问题与知识库文档可能相关时，优先使用 knowledge_search 检索；
- 当前知识库中已有文档：{doc_list}
- 回答使用中文，简洁、准确；引用文档内容时注明来源文件名。"""


def build_agent_executor() -> AgentExecutor:
    """每次请求构建 Agent（系统提示中注入最新文档列表）。"""
    system_text = SYSTEM_TEMPLATE.format(
        date=date.today().isoformat(),
        doc_list="、".join(list_documents()) or "（空）",
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=get_llm(streaming=True), tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=8,
    )


async def stream_chat(session: Session, question: str):
    """流式执行 Agent，逐个产出 SSE 事件（token / 工具调用 / 错误 / 结束）。"""
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        agent_executor = build_agent_executor()
        inputs = {"input": question, "chat_history": store.windowed_history(session)}
        collected: list[str] = []
        final_output = None
        async for ev in agent_executor.astream_events(inputs, version="v2"):
            kind = ev["event"]
            if kind == "on_chat_model_stream":
                content = ev["data"]["chunk"].content
                if isinstance(content, str) and content:
                    collected.append(content)
                    yield _sse({"type": "token", "content": content})
            elif kind == "on_tool_start":
                yield _sse({
                    "type": "tool_start",
                    "name": ev["name"],
                    "input": json.dumps(ev["data"].get("input", {}), ensure_ascii=False),
                })
            elif kind == "on_tool_end":
                out = ev["data"].get("output")
                text = getattr(out, "content", None) or str(out)
                yield _sse({"type": "tool_end", "name": ev["name"], "output": str(text)[:300]})
            elif kind == "on_chain_end" and ev.get("name") == "AgentExecutor":
                out = ev["data"].get("output")
                if isinstance(out, dict) and out.get("output"):
                    final_output = out["output"]
    except Exception as e:
        yield _sse({"type": "error", "content": f"{type(e).__name__}: {e}"})
        return

    answer = final_output or "".join(collected)
    if answer.strip():
        store.append(session.id, [HumanMessage(content=question), AIMessage(content=answer)])
    yield _sse({"type": "done", "content": answer})

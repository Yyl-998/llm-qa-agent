"""知识库工具：RAG 语义检索 + 文档摘要。"""
from langchain_core.tools import tool

from backend.config import ALLOWED_EXTS, DOCS_DIR
from backend.llm import get_llm
from backend.rag.knowledge import knowledge_base
from backend.utils.loader import load_document, split_documents


def list_documents() -> list[str]:
    """返回知识库文档名列表，按上传时间倒序（最新在前）。"""
    files = [p for p in DOCS_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTS]
    return [p.name for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]


@tool
def knowledge_search(query: str) -> str:
    """在用户上传的知识库文档中进行语义检索（RAG）。当用户的问题可能与已上传的文档有关时调用。"""
    try:
        if knowledge_base.is_empty():
            docs = list_documents()
            if not docs:
                return "知识库为空，暂未上传任何文档。"
            return f"知识库已有文档（{'、'.join(docs)}）但尚未建立向量索引，请提示用户在界面重新上传。"
        chunks = knowledge_base.search(query, k=4)
        if not chunks:
            return "知识库中未检索到与问题相关的内容。"
        parts = [
            f"[片段{i} | 来源: {c.metadata.get('source', '未知')}]\n{c.page_content}"
            for i, c in enumerate(chunks, 1)
        ]
        return "以下是从知识库中检索到的相关内容：\n\n" + "\n\n".join(parts)
    except Exception as e:
        return f"知识库检索失败：{e}"


@tool
def summarize_document(file_name: str = "") -> str:
    """对知识库中的指定文档进行内容摘要总结。file_name 为文档文件名；留空则自动选择最近上传的文档。"""
    try:
        files = list_documents()
        if not files:
            return "知识库为空，请先上传文档。"
        if file_name not in files:
            file_name = files[0]  # 默认摘要最近上传的文档
        path = DOCS_DIR / file_name
        docs = load_document(path)
        text = "\n".join(d.page_content for d in docs)
        if not text.strip():
            return f"文档《{file_name}》内容为空或无法解析。"

        llm = get_llm(streaming=False)
        if len(text) <= 3000:
            prompt = f"请用中文对以下文档内容做一份结构化摘要（要点式，包含核心结论）：\n\n{text[:12000]}"
            return llm.invoke(prompt).content

        # 长文档：Map-Reduce 摘要（最多取前 2 万字符、10 个片段）
        pieces = [text[i:i + 2000] for i in range(0, min(len(text), 20000), 2000)][:10]
        partials = [
            llm.invoke(f"请用中文用 3 句话概括以下内容片段：\n\n{p}").content
            for p in pieces
        ]
        joined = "\n".join(f"- {p}" for p in partials)
        return llm.invoke(
            f"以下是一篇文档《{file_name}》各部分的要点概括，请整合成一份完整的结构化中文摘要：\n\n{joined}"
        ).content
    except Exception as e:
        return f"文档摘要失败：{e}"

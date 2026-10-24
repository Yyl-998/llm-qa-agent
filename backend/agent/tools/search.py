"""网页搜索工具：基于 DuckDuckGo（ddgs），无需 API Key。"""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """搜索互联网获取实时信息、新闻或你不确定的知识。输入为搜索关键词。"""
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # 兼容旧包名
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="cn-zh", max_results=5))
        if not results:
            return f"没有找到与「{query}」相关的搜索结果。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r.get('title', '')}\n"
                f"   {r.get('body', r.get('snippet', ''))}\n"
                f"   来源: {r.get('href', r.get('url', ''))}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"搜索失败：{e}（如网络受限请稍后重试）"

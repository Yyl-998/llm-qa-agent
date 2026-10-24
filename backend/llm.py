"""LLM 客户端工厂（OpenAI 兼容协议，默认 DeepSeek）。"""
from functools import lru_cache

from backend.config import LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE


@lru_cache(maxsize=4)
def get_llm(streaming: bool = True):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
        streaming=streaming,
        timeout=120,
        max_retries=2,
    )

"""全局配置：从项目根目录 .env 读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# 大模型（OpenAI 兼容协议，默认 DeepSeek）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Embedding：EMBEDDING_MODEL 为空时回退到本地 fastembed 模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENAI_BASE_URL)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_LOCAL_MODEL = os.getenv("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")

# HuggingFace 镜像（国内网络下载本地 embedding 模型用，须在加载 huggingface_hub 前设置）
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

# 会话记忆：保留最近 N 轮对话作为上下文
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# 知识库目录
# 原始文档放 AppData\Local：火绒等杀软的"文档防护"会拦截网络服务进程（uvicorn）
# 写入 Documents 下的 PDF/DOCX（PermissionError），AppData 不在其守护范围
DOCS_DIR = Path(os.environ.get("LOCALAPPDATA", str(BASE_DIR))) / "llm-qa-agent" / "documents"
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".docx"}

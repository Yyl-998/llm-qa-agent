"""RAG 知识库：FAISS 向量检索，索引持久化到 data/vectorstore。

Embedding 策略：
- 配置了 EMBEDDING_MODEL -> OpenAI 兼容远程 embedding 接口
- 未配置 -> 本地 fastembed（bge-small-zh 中文模型，无需 API Key）
"""
import threading

from backend.config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_LOCAL_MODEL, EMBEDDING_MODEL, VECTOR_DIR

_lock = threading.Lock()


def get_embeddings():
    if EMBEDDING_MODEL:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY or None,
            base_url=EMBEDDING_BASE_URL or None,
        )
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=EMBEDDING_LOCAL_MODEL)


class KnowledgeBase:
    def __init__(self):
        self._db = None

    def is_empty(self) -> bool:
        """索引文件是否存在（不触发 embedding 模型加载）。"""
        return not (VECTOR_DIR / "index.faiss").exists()

    def _ensure_db(self):
        if self._db is None and not self.is_empty():
            from langchain_community.vectorstores import FAISS

            self._db = FAISS.load_local(
                str(VECTOR_DIR), get_embeddings(), allow_dangerous_deserialization=True
            )
        return self._db

    def add_document(self, chunks) -> int:
        """文档片段入库并向量化，返回片段数。"""
        with _lock:
            from langchain_community.vectorstores import FAISS

            db = self._ensure_db()
            if db is None:
                db = FAISS.from_documents(chunks, get_embeddings())
            else:
                db.add_documents(chunks)
            db.save_local(str(VECTOR_DIR))
            self._db = db
            return len(chunks)

    def search(self, query: str, k: int = 4):
        db = self._ensure_db()
        if db is None:
            return []
        return db.similarity_search(query, k=k)

    def remove_document(self, source: str) -> int:
        """按来源文件名删除对应向量，返回删除的片段数。"""
        with _lock:
            db = self._ensure_db()
            if db is None:
                return 0
            ids = [
                key
                for key, doc in db.docstore._dict.items()
                if doc.metadata.get("source") == source
            ]
            if not ids:
                return 0
            db.delete(ids)
            if db.index.ntotal == 0:
                # 索引已空：删除索引文件，回到初始状态
                self._db = None
                for f in ("index.faiss", "index.pkl"):
                    (VECTOR_DIR / f).unlink(missing_ok=True)
            else:
                db.save_local(str(VECTOR_DIR))
            return len(ids)


knowledge_base = KnowledgeBase()

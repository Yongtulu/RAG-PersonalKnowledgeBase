"""
ChromaDB 向量存储操作 — 初始化、添加文档、按来源删除
"""
from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


_CHROMA_BATCH = 500   # 单次写入上限，留余量低于 ChromaDB 的 5461

def add_documents(docs: list) -> None:
    if not docs:
        return
    vs = get_vectorstore()
    for i in range(0, len(docs), _CHROMA_BATCH):
        vs.add_documents(docs[i : i + _CHROMA_BATCH])
    # CLI 路径使用此函数；webui 直接调用 get_vectorstore() 以便 yield 进度


def delete_by_source(source_path: str) -> int:
    """按文件路径删除所有相关 chunks，返回删除数量。"""
    vs = get_vectorstore()
    collection = vs._collection

    # 精确路径匹配
    results = collection.get(where={"source": source_path}, include=["metadatas"])
    ids = results.get("ids", [])

    # 兜底：按文件名匹配（分页拉取，避免 SQL 变量超限）
    if not ids:
        filename = Path(source_path).name
        ids = []
        offset = 0
        batch = 500
        while True:
            chunk = collection.get(limit=batch, offset=offset, include=["metadatas"])
            if not chunk["ids"]:
                break
            for doc_id, meta in zip(chunk["ids"], chunk["metadatas"]):
                if Path(meta.get("source", "")).name == filename:
                    ids.append(doc_id)
            if len(chunk["ids"]) < batch:
                break
            offset += batch

    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_sources() -> list[dict]:
    """从注册表读取已索引文件列表，避免全量查询 ChromaDB 触发 SQL 变量超限。"""
    from src.registry import load_registry
    registry = load_registry()
    return [{"source": src, "name": Path(src).name} for src in registry]


def get_retriever(top_k: int = None):
    from src.config import TOP_K
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k or TOP_K})

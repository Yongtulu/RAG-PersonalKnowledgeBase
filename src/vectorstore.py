"""
ChromaDB 向量存储操作模块
─────────────────────────
负责与向量数据库交互：
  - 初始化 Embedding 模型（文字 → 向量）
  - 初始化 ChromaDB 连接
  - 添加文档（写入向量）
  - 按来源删除文档
  - 查询已索引文件列表
  - 构建检索器（供 RAG 链使用）

ChromaDB 本地持久化：所有向量和原文都存储在 chroma_db/ 目录，
程序重启后数据不丢失，无需重新索引。
"""
from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

# 全局单例：Embedding 模型只加载一次，避免重复占用内存和时间
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    懒加载 Embedding 模型（首次调用时才下载/加载）。
    paraphrase-multilingual-MiniLM-L12-v2 输出 384 维向量，
    normalize_embeddings=True 使向量长度归一化为 1，
    这样余弦相似度等价于点积，计算更快。
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},          # 使用 CPU 推理
            encode_kwargs={"normalize_embeddings": True},  # 归一化，便于余弦相似度计算
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """
    创建并返回 ChromaDB 向量库实例。
    persist_directory 指定持久化路径，数据写入后会自动保存到磁盘。
    每次调用都会创建新实例，但底层连接的是同一个持久化目录，数据共享。
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,     # 集合名，类似数据库的表名
        embedding_function=get_embeddings(), # 插入/查询时自动调用此函数做向量化
        persist_directory=str(CHROMA_DIR),   # 数据持久化目录
    )


# ChromaDB 单次 upsert 的最大条数上限（官方限制约 5461，留余量取 500）
# upsert = update + insert：已存在则更新，不存在则插入
_CHROMA_BATCH = 500


def add_documents(docs: list) -> None:
    """
    将文档块批量写入向量库。
    分批写入（每批 500 条）避免触发 ChromaDB 的单次批量上限。
    CLI 的 ingest 命令通过此函数写入；
    Web UI 直接调用 get_vectorstore() 以便在批次之间 yield 进度。
    """
    if not docs:
        return
    vs = get_vectorstore()
    for i in range(0, len(docs), _CHROMA_BATCH):
        vs.add_documents(docs[i : i + _CHROMA_BATCH])


def delete_by_source(source_path: str) -> int:
    """
    删除某个文件对应的所有向量块，返回实际删除的数量。

    删除策略（两步兜底）：
      1. 精确匹配：用 where={"source": source_path} 过滤，直接拿 ID
      2. 文件名匹配：分页扫描（每页 500 条），按文件名对比
         ——应对路径格式不一致的情况（如相对路径 vs 绝对路径）

    分页原因：collection.get() 不加 limit 会拉取全表，
    数据量大时会触发 SQLite "too many SQL variables" 错误。
    """
    vs = get_vectorstore()
    collection = vs._collection

    # 第一步：精确路径匹配（include 只取 metadatas，不拉 embeddings，节省内存）
    results = collection.get(where={"source": source_path}, include=["metadatas"])
    ids = results.get("ids", [])

    # 第二步：按文件名分页扫描（兜底）
    if not ids:
        filename = Path(source_path).name
        ids = []
        offset = 0
        batch = 500
        while True:
            chunk = collection.get(limit=batch, offset=offset, include=["metadatas"])
            if not chunk["ids"]:
                break  # 没有更多数据了
            for doc_id, meta in zip(chunk["ids"], chunk["metadatas"]):
                if Path(meta.get("source", "")).name == filename:
                    ids.append(doc_id)
            if len(chunk["ids"]) < batch:
                break  # 最后一页，扫描完毕
            offset += batch

    if ids:
        collection.delete(ids=ids)
    return len(ids)


def list_sources() -> list[dict]:
    """
    返回所有已索引文件的信息列表。
    直接读注册表文件（JSON），不查询 ChromaDB，
    避免全量拉取触发 SQL 变量超限，同时速度更快。
    """
    from src.registry import load_registry
    registry = load_registry()
    return [{"source": src, "name": Path(src).name} for src in registry]


def get_retriever(top_k: int = None):
    """
    构建向量相似度检索器。
    检索时：将查询文本向量化 → 在向量库中找最近的 top_k 个块 → 返回原文。
    相似度算法默认为余弦相似度（因为向量已归一化）。
    """
    from src.config import TOP_K
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k or TOP_K})

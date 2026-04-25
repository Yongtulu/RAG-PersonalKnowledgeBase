"""
文档加载与切分模块
─────────────────
负责两件事：
  1. 把磁盘上的文件（PDF / Markdown / TXT）读成 LangChain Document 对象
  2. 把长文档切成适合 Embedding 的小块（chunk）

LangChain Document 结构：
  Document(
      page_content = "文本内容",
      metadata     = {"source": "/path/to/file.pdf", "page": 3}
  )
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_file(path: Path) -> list:
    """
    加载单个文件，返回 Document 列表。

    PDF 文件：使用 PyPDFLoader，每页生成一个 Document，
              metadata 中会自动带上页码 {"page": 0, "source": "..."}
    其他文件：使用 TextLoader，整个文件作为一个 Document，
              autodetect_encoding=True 自动识别 GBK / UTF-8 等编码
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        # autodetect_encoding 会尝试多种编码，避免中文乱码报错
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def load_files(paths: list[Path]) -> list:
    """
    批量加载多个文件，返回合并后的 Document 列表。
    单个文件加载失败时打印警告并跳过，不影响其他文件。
    """
    docs = []
    for p in paths:
        try:
            docs.extend(load_file(p))
        except Exception as e:
            print(f"[warn] 跳过 {p.name}：{e}")
    return docs


def split_documents(docs: list) -> list:
    """
    将 Document 列表切分为更小的块（chunk）。

    使用 RecursiveCharacterTextSplitter：
      - 优先按段落（\\n\\n）切分，保持语义完整
      - 段落不够小时按行（\\n）切分
      - 再不够小就按句子标点切分
      - 最后按空格、单字符兜底

    separators 列表的顺序就是优先级，越靠前越优先尝试。
    中文标点（。！？）放在英文标点（.!?）前面，对中文文档更友好。

    chunk_size    = 512  字符：单块不超过 512 字符
    chunk_overlap = 64   字符：相邻块共享 64 字符，防止句子在块边界处被截断
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)

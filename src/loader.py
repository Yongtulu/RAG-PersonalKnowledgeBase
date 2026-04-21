"""
文档加载与切分 — 支持 PDF / Markdown / TXT
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_file(path: Path) -> list:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    return loader.load()


def load_files(paths: list[Path]) -> list:
    docs = []
    for p in paths:
        try:
            docs.extend(load_file(p))
        except Exception as e:
            print(f"[warn] 跳过 {p.name}：{e}")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)

"""
增量索引注册表
─────────────
核心思路：用 MD5 哈希值追踪每个文件的内容指纹。
每次 ingest 前，先比对文件的当前哈希和注册表里记录的哈希：
  - 哈希相同 → 文件未变化 → 跳过，不重复向量化
  - 哈希不同或不存在 → 新文件或已修改 → 需要处理

注册表以 JSON 格式保存在 chroma_db/file_registry.json：
{
  "/absolute/path/to/file.pdf": "abc123def456...",  # MD5 哈希
  ...
}
"""
import hashlib
import json
from pathlib import Path

from src.config import DOCS_DIR, REGISTRY_FILE

# 支持的文档格式，其他格式会被跳过
SUPPORTED_EXTS = {".pdf", ".md", ".txt", ".rst"}


def _md5(path: Path) -> str:
    """
    计算文件的 MD5 哈希值，用作内容指纹。
    分块读取（8KB/次），避免大文件一次性占用过多内存。
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        # iter(callable, sentinel)：每次调用 lambda 读 8192 字节，直到返回 b""
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    """
    从磁盘加载注册表。
    如果文件不存在（首次运行），返回空字典。
    """
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


def save_registry(registry: dict) -> None:
    """
    将注册表写回磁盘。
    ensure_ascii=False 保证中文路径正常保存（不被转义成 \\uXXXX）。
    """
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def get_all_doc_files(docs_dir: Path = DOCS_DIR) -> list[Path]:
    """
    递归扫描目录，返回所有支持格式的文件路径列表。
    rglob("*") 会递归进入所有子目录。
    """
    return [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def get_new_files(docs_dir: Path = DOCS_DIR, registry: dict = None) -> list[Path]:
    """
    返回需要（重新）索引的文件列表：
      - 注册表中没有记录的文件（新文件）
      - 注册表中有记录但 MD5 已变化的文件（内容被修改过）
    """
    if registry is None:
        registry = load_registry()
    result = []
    for path in get_all_doc_files(docs_dir):
        key = str(path)
        # 文件不在注册表 OR 哈希值不匹配 → 需要处理
        if key not in registry or registry[key] != _md5(path):
            result.append(path)
    return result


def register_files(files: list[Path], registry: dict) -> dict:
    """
    将已处理的文件及其 MD5 写入注册表（内存中）。
    调用后还需要调用 save_registry() 才会持久化到磁盘。
    """
    for path in files:
        registry[str(path)] = _md5(path)
    return registry


def unregister_file(filepath: str, registry: dict) -> dict:
    """
    从注册表中移除一个文件的记录。
    支持两种匹配方式：
      1. 完整路径精确匹配（精确）
      2. 仅文件名匹配（兜底，应对路径格式不一致的情况）
    """
    # 先尝试精确路径匹配
    registry.pop(filepath, None)
    # 再尝试按文件名匹配（路径可能有差异）
    to_remove = [k for k in registry if Path(k).name == filepath]
    for k in to_remove:
        registry.pop(k)
    return registry

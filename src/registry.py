"""
增量索引注册表 — 用 MD5 哈希追踪已处理文件，避免重复向量化
"""
import hashlib
import json
from pathlib import Path

from src.config import DOCS_DIR, REGISTRY_FILE

SUPPORTED_EXTS = {".pdf", ".md", ".txt", ".rst"}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def get_all_doc_files(docs_dir: Path = DOCS_DIR) -> list[Path]:
    return [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def get_new_files(docs_dir: Path = DOCS_DIR, registry: dict = None) -> list[Path]:
    """返回新增或内容有变化的文件列表。"""
    if registry is None:
        registry = load_registry()
    result = []
    for path in get_all_doc_files(docs_dir):
        key = str(path)
        if key not in registry or registry[key] != _md5(path):
            result.append(path)
    return result


def register_files(files: list[Path], registry: dict) -> dict:
    for path in files:
        registry[str(path)] = _md5(path)
    return registry


def unregister_file(filepath: str, registry: dict) -> dict:
    registry.pop(filepath, None)
    # also try matching by filename only
    to_remove = [k for k in registry if Path(k).name == filepath]
    for k in to_remove:
        registry.pop(k)
    return registry

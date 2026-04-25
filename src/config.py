"""
全局配置模块
所有路径、模型名称、超参数都在这里统一管理。
其他模块通过 from src.config import XXX 取用，不要在代码里写死字符串。
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（如果存在）
# 优先级：.env 文件 < 系统环境变量
load_dotenv()

# ── 路径 ──────────────────────────────────────────────────────────────────────

# 项目根目录（config.py 在 src/ 下，所以往上一级）
BASE_DIR = Path(__file__).parent.parent

# 用户文档目录：把 PDF / Markdown / TXT 放在这里
DOCS_DIR = BASE_DIR / "docs"

# ChromaDB 持久化目录：向量索引和注册表都存在这里
CHROMA_DIR = BASE_DIR / "chroma_db"

# 增量索引注册表：记录每个文件的 MD5，用于判断文件是否变动
REGISTRY_FILE = CHROMA_DIR / "file_registry.json"

# ── Embedding 模型 ────────────────────────────────────────────────────────────

# 多语言 Sentence-BERT，支持中英文混合
# 输出 384 维向量，模型体积约 118MB，首次运行时自动从 HuggingFace 下载
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# ── LLM（大语言模型）─────────────────────────────────────────────────────────

# Ollama 本地服务的模型名称，需要提前运行 ollama pull <model>
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")

# Ollama HTTP 服务地址，默认本机 11434 端口
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── 检索参数 ──────────────────────────────────────────────────────────────────

# 每次检索返回的最相关文本块数量
# 数字越大上下文越丰富，但也会增加 LLM 的输入长度
TOP_K = int(os.getenv("TOP_K", "5"))

# 文档切块大小（字符数）
# 太大：单块语义混杂，检索精度下降；太小：可能切断完整句子
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))

# 相邻块之间的重叠字符数，防止句子被切断导致语义断裂
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ChromaDB 中的集合名称（类似数据库里的表名）
COLLECTION_NAME = "knowledge_base"

"""
RAG 问答链模块
──────────────
使用纯 LangChain Core（LCEL）实现，不依赖 langchain.chains（高版本已移除）。

整体流程：
  用户问题
    ↓（有历史时）CONDENSE_PROMPT → LLM → 改写为独立问题
    ↓ 检索器 → 从向量库取最相关的 TOP_K 个文本块
    ↓ QA_PROMPT（上下文 + 问题）→ LLM → 生成答案

LCEL（LangChain Expression Language）管道语法：
  A | B | C  等价于  C(B(A(input)))
  用 | 把各个处理步骤串联，数据从左往右流动。
"""
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.prompts import CONDENSE_PROMPT, QA_PROMPT
from src.vectorstore import get_retriever

# 全局单例：LLM 只初始化一次，避免每次提问都重新建立连接
_llm = None


def get_llm() -> ChatOllama:
    """
    懒加载 Ollama LLM 实例。
    temperature=0.1：接近 0 表示输出更稳定确定，减少随机发挥。
    （temperature=0 完全确定，temperature=1 创意最强但容易跑偏）
    """
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
    return _llm


def build_rag_chain():
    """
    构建完整的 RAG 问答链，返回一个可调用的 Runnable 对象。

    链的内部逻辑（_run 函数）：
      Step 1 - 问题改写（仅多轮时需要）
        有历史：condense_chain = CONDENSE_PROMPT | LLM | StrOutputParser
                把"它的缺点是什么" + 历史 → "逻辑回归的缺点是什么"
        无历史：直接使用原始问题

      Step 2 - 向量检索
        把改写后的独立问题送入向量检索器，
        返回 TOP_K 个最相关的 Document 对象

      Step 3 - 生成答案
        answer_chain = QA_PROMPT | LLM | StrOutputParser
        把检索到的文本块拼成 context，连同问题和历史一起喂给 LLM

    返回值格式：{"answer": "...", "context": [Document, ...]}
    """
    llm = get_llm()
    retriever = get_retriever()

    # StrOutputParser：把 LLM 返回的 AIMessage 对象 → 纯字符串
    str_out = StrOutputParser()

    # 问题压缩链：有历史时用，把追问改写为独立问题
    condense_chain = CONDENSE_PROMPT | llm | str_out

    # 答案生成链：拿到上下文后生成最终答案
    answer_chain = QA_PROMPT | llm | str_out

    def _run(inputs: dict) -> dict:
        question = inputs["input"]
        history = inputs.get("chat_history", [])

        # Step 1：有对话历史时，先把问题改写为独立问题
        # 例："它的缺点？" + 历史["我们在讨论逻辑回归"] → "逻辑回归的缺点是什么？"
        if history:
            standalone = condense_chain.invoke({
                "input": question,
                "chat_history": history,
            })
        else:
            standalone = question  # 首轮问题直接用

        # Step 2：用独立问题去向量库检索最相关的文本块
        # retriever 内部：问题向量化 → 余弦相似度计算 → 返回 TOP_K 个 Document
        docs = retriever.invoke(standalone)

        # Step 3：把检索到的所有文本块拼接成上下文字符串
        context_str = "\n\n".join(doc.page_content for doc in docs)

        # Step 4：把上下文 + 问题 + 历史 交给 LLM 生成答案
        answer = answer_chain.invoke({
            "input": question,       # 原始问题（不是改写后的，保持用户原意）
            "chat_history": history,
            "context": context_str,  # 检索到的参考资料
        })

        # context 保留原始 Document 列表，供调用方提取来源信息
        return {"answer": answer, "context": docs}

    # 用 RunnableLambda 包装普通函数，使其融入 LCEL 管道
    return RunnableLambda(_run)


def ask_once(question: str) -> dict:
    """
    单次问答（无历史），返回 {"answer": str, "sources": list}。
    适合 CLI 的 ask 命令。
    """
    chain = build_rag_chain()
    result = chain.invoke({"input": question, "chat_history": []})
    sources = _extract_sources(result.get("context", []))
    return {"answer": result["answer"], "sources": sources}


def ask_with_history(question: str, chat_history: list, chain=None) -> tuple:
    """
    带历史记忆的问答，返回 (answer, updated_history, sources)。
    适合多轮对话（CLI 的 chat 命令 和 Web UI）。

    chat_history 格式：[HumanMessage(...), AIMessage(...), HumanMessage(...), ...]
    每轮对话后把新的一问一答追加进去，传给下一轮使用。
    """
    if chain is None:
        chain = build_rag_chain()

    result = chain.invoke({"input": question, "chat_history": chat_history})
    answer = result["answer"]

    # 把本轮对话追加到历史记录，传给下一轮
    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
    sources = _extract_sources(result.get("context", []))
    return answer, updated_history, sources


def _extract_sources(context_docs: list) -> list[dict]:
    """
    从检索到的 Document 列表中提取来源信息，去重后返回。
    每个来源包含：文件名、页码、内容片段（前 150 字）。
    """
    seen = set()
    sources = []
    for doc in context_docs:
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "-")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": Path(src).name if src else "未知",
                "page": page,
                "snippet": doc.page_content[:150].replace("\n", " "),
            })
    return sources

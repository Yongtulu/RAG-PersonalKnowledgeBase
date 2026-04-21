"""
LangChain RAG 链 — 单次问答 + 多轮对话（带历史记忆）
纯 langchain-core LCEL 实现，不依赖 langchain.chains
"""
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.prompts import CONDENSE_PROMPT, QA_PROMPT
from src.vectorstore import get_retriever

_llm = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
    return _llm


def build_rag_chain():
    """纯 LCEL 实现：history-aware 检索 + 问答，返回 {answer, context}。"""
    llm = get_llm()
    retriever = get_retriever()
    str_out = StrOutputParser()
    condense_chain = CONDENSE_PROMPT | llm | str_out
    answer_chain = QA_PROMPT | llm | str_out

    def _run(inputs: dict) -> dict:
        question = inputs["input"]
        history = inputs.get("chat_history", [])

        # 有历史时先压缩为独立问题
        if history:
            standalone = condense_chain.invoke({"input": question, "chat_history": history})
        else:
            standalone = question

        docs = retriever.invoke(standalone)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        answer = answer_chain.invoke({
            "input": question,
            "chat_history": history,
            "context": context_str,
        })
        return {"answer": answer, "context": docs}

    return RunnableLambda(_run)


def ask_once(question: str) -> dict:
    """单次问答，无历史上下文。"""
    chain = build_rag_chain()
    result = chain.invoke({"input": question, "chat_history": []})
    sources = _extract_sources(result.get("context", []))
    return {"answer": result["answer"], "sources": sources}


def ask_with_history(question: str, chat_history: list, chain=None) -> tuple[str, list]:
    """带历史的问答，返回 (answer, updated_history)。"""
    if chain is None:
        chain = build_rag_chain()
    result = chain.invoke({"input": question, "chat_history": chat_history})
    answer = result["answer"]
    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
    sources = _extract_sources(result.get("context", []))
    return answer, updated_history, sources


def _extract_sources(context_docs: list) -> list[dict]:
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

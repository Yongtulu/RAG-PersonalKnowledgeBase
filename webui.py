#!/usr/bin/env python3
"""
个人知识库 Web UI — 基于 Gradio
运行：python webui.py
访问：http://localhost:7860
"""
from pathlib import Path

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

from src.config import DOCS_DIR
from src.registry import _md5

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── 状态 ──────────────────────────────────────────────────────────────────────

_chain = None
_chat_history = []  # LangChain message 列表


def _build_display(override_last: str = None) -> list:
    """从 _chat_history 重建 Gradio 6.x messages 格式：[{"role":..,"content":..}, ...]"""
    display = []
    for msg in _chat_history:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            display.append({"role": "assistant", "content": msg.content})
    if override_last and display and display[-1]["role"] == "assistant":
        display[-1]["content"] = override_last
    return display


def _get_chain():
    global _chain
    if _chain is None:
        from src.chains import build_rag_chain
        _chain = build_rag_chain()
    return _chain


def _reset_chain():
    global _chain, _chat_history
    _chain = None
    _chat_history = []


# ── 目录索引 ──────────────────────────────────────────────────────────────────

def ingest_any(dir_path: str, uploaded_files, force: bool):
    """同时支持目录路径 + 上传文件；逐文件处理并 yield 实时进度。"""
    from src.registry import (
        load_registry, get_all_doc_files,
        register_files, save_registry, SUPPORTED_EXTS,
    )
    from src.loader import load_files, split_documents
    from src.vectorstore import add_documents

    candidate_files: list[Path] = []
    errors: list[str] = []

    # ── 目录或文件路径 ──
    dir_path = (dir_path or "").strip()
    if dir_path:
        target = Path(dir_path).expanduser().resolve()
        if not target.exists():
            errors.append(f"路径不存在：{target}")
        elif target.is_file():
            if target.suffix.lower() in SUPPORTED_EXTS:
                candidate_files.append(target)
            else:
                errors.append(f"不支持的文件类型：{target.name}")
        elif target.is_dir():
            candidate_files.extend(get_all_doc_files(target))

    # ── 上传文件 ──
    for f in (uploaded_files or []):
        p = Path(f.name)
        if p.suffix.lower() in SUPPORTED_EXTS:
            candidate_files.append(p)
        else:
            errors.append(f"跳过不支持的文件：{p.name}")

    if not candidate_files and not errors:
        yield "请输入目录路径或上传文件。", _get_doc_table()
        return
    if not candidate_files:
        yield "\n".join(errors), _get_doc_table()
        return

    registry = {} if force else load_registry()
    new_files = [
        p for p in candidate_files
        if force or registry.get(str(p)) != _md5(p)
    ]

    prefix = ("\n".join(errors) + "\n") if errors else ""

    if not new_files:
        yield prefix + f"✓ 共 {len(candidate_files)} 个文件，全部已是最新，无需重新索引。", _get_doc_table()
        return

    total = len(new_files)
    yield prefix + f"共 {total} 个文件待处理，开始索引...", _get_doc_table()

    from src.vectorstore import get_vectorstore, _CHROMA_BATCH

    total_chunks = 0
    for i, file in enumerate(new_files, 1):
        yield prefix + f"[{i}/{total}] 解析中：{file.name}", _get_doc_table()
        docs = load_files([file])
        if not docs:
            yield prefix + f"[{i}/{total}] ⚠ 跳过（解析失败）：{file.name}", _get_doc_table()
            continue
        chunks = split_documents(docs)
        if not chunks:
            yield prefix + f"[{i}/{total}] ⚠ 跳过（内容为空）：{file.name}", _get_doc_table()
            continue

        n = len(chunks)
        n_batches = (n + _CHROMA_BATCH - 1) // _CHROMA_BATCH
        vs = get_vectorstore()
        for b, start in enumerate(range(0, n, _CHROMA_BATCH), 1):
            batch = chunks[start : start + _CHROMA_BATCH]
            vs.add_documents(batch)
            yield (
                prefix + f"[{i}/{total}] 向量化 {file.name}：{min(start + _CHROMA_BATCH, n)}/{n} 块"
                + (f"（第 {b}/{n_batches} 批）" if n_batches > 1 else ""),
                _get_doc_table(),
            )

        register_files([file], registry)
        save_registry(registry)
        total_chunks += n

    _reset_chain()
    yield prefix + f"✓ 全部完成：{total} 个文件，{total_chunks} 个文本块", _get_doc_table()


# ── 文档列表 ──────────────────────────────────────────────────────────────────

def _get_doc_table() -> str:
    from src.vectorstore import list_sources
    sources = list_sources()
    if not sources:
        return "（暂无已索引文件）"
    lines = [f"[{i+1}] {s['name']}\n    {s['source']}" for i, s in enumerate(sources)]
    return "\n\n".join(lines)


def refresh_docs():
    return _get_doc_table()


def delete_doc(selected_name: str):
    if not selected_name or not selected_name.strip():
        return "请输入要删除的文件名。", _get_doc_table()

    from src.vectorstore import delete_by_source, list_sources
    from src.registry import load_registry, unregister_file, save_registry

    sources = list_sources()
    match = next((s for s in sources if s["name"] == selected_name.strip()), None)
    if not match:
        return f"未找到文件 '{selected_name}'。", _get_doc_table()

    count = delete_by_source(match["source"])
    registry = load_registry()
    unregister_file(match["source"], registry)
    save_registry(registry)
    _reset_chain()

    return f"✓ 已删除 '{match['name']}' 的 {count} 个文本块。", _get_doc_table()


# ── 聊天 ─────────────────────────────────────────────────────────────────────

def chat(user_message: str, history: list):
    """重建显示历史，不拼接传入的 history，避免 Gradio 版本间的类型混用问题。"""
    global _chat_history

    if not user_message.strip():
        return _build_display(), ""

    from src.chains import ask_with_history
    try:
        answer, _chat_history, sources = ask_with_history(
            user_message, _chat_history, _get_chain()
        )
    except Exception as e:
        answer = f"[错误] {e}\n\n请确认 Ollama 服务已启动，并运行过 `python app.py ingest`。"
        _chat_history = _chat_history + [
            HumanMessage(content=user_message),
            AIMessage(content=answer),
        ]
        sources = []

    if sources:
        src_lines = "\n".join(
            f"- {s['file']} 第{s['page']}页：{s['snippet']}..."
            for s in sources
        )
        answer += f"\n\n---\n**参考来源**\n{src_lines}"

    return _build_display(override_last=answer), ""


def clear_history():
    global _chat_history
    _chat_history = []
    return [], ""


# ── UI 构建 ───────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="个人知识库问答") as demo:
        gr.Markdown("# 个人知识库问答系统\n基于 LangChain + ChromaDB + Ollama (gemma4:31b)")

        with gr.Row():
            # 左侧：文档管理
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## 文档管理")

                dir_input = gr.Textbox(
                    label="目录或文件路径（支持 ~ 展开）",
                    placeholder="/Users/jungang/Documents/课程笔记",
                    lines=1,
                )
                file_upload = gr.File(
                    label="或直接上传文件（PDF / Markdown / TXT，可多选）",
                    file_types=[".pdf", ".md", ".txt", ".rst"],
                    file_count="multiple",
                )
                force_checkbox = gr.Checkbox(label="强制重建（忽略缓存，重新索引所有文件）", value=False)
                ingest_btn = gr.Button("建立索引", variant="primary")
                ingest_status = gr.Textbox(label="进度", interactive=False, lines=3)

                gr.Markdown("### 已索引文件")
                doc_table = gr.Textbox(
                    value=_get_doc_table(),
                    lines=8,
                    max_lines=30,
                    interactive=False,
                )
                refresh_btn = gr.Button("刷新列表")

                gr.Markdown("### 删除文件索引")
                delete_input = gr.Textbox(
                    label="输入文件名（如 lecture1.pdf）",
                    placeholder="lecture1.pdf",
                )
                delete_btn = gr.Button("删除", variant="stop")
                delete_status = gr.Textbox(label="状态", interactive=False)

            # 右侧：聊天
            with gr.Column(scale=2):
                gr.Markdown("## 知识库问答")
                chatbot = gr.Chatbot(height=520, label="对话")
                msg_input = gr.Textbox(
                    placeholder="输入问题，按 Enter 发送...",
                    label="",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空对话")

        # ── 事件绑定 ──────────────────────────────────────────────────────────

        ingest_btn.click(
            ingest_any,
            inputs=[dir_input, file_upload, force_checkbox],
            outputs=[ingest_status, doc_table],
        )

        refresh_btn.click(refresh_docs, outputs=[doc_table])

        delete_btn.click(
            delete_doc,
            inputs=[delete_input],
            outputs=[delete_status, doc_table],
        )

        send_btn.click(
            chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        clear_btn.click(clear_history, outputs=[chatbot, msg_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())

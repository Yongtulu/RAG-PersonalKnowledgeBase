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
from src.registry import _md5  # 用于判断文件是否变更

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ── 全局状态 ──────────────────────────────────────────────────────────────────
# 这两个变量在整个 Web 会话期间保持，不随请求重置。
# 多用户场景下需要改成 session 级别，但个人知识库单用户够用。

_chain = None        # RAG 链实例（懒加载，首次提问时初始化）
_chat_history = []   # LangChain 格式的对话历史：[HumanMessage, AIMessage, ...]


def _build_display(override_last: str = None) -> list:
    """
    把内部的 LangChain 消息列表转换成 Gradio Chatbot 组件要求的格式：
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    为什么不直接用 Gradio 传入的 history 参数拼接？
    因为不同 Gradio 版本传入的 history 格式不一致（有时是 dict，有时是 ChatMessage 对象），
    直接拼接会产生混合类型列表，触发格式校验错误。
    改为从 _chat_history 重建，输出始终是纯 dict，格式稳定。

    override_last：用来把最后一条助手消息替换为带来源引用的完整版本。
    """
    display = []
    for msg in _chat_history:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            display.append({"role": "assistant", "content": msg.content})
    # 如果需要，把最后一条助手消息替换为附带来源的版本
    if override_last and display and display[-1]["role"] == "assistant":
        display[-1]["content"] = override_last
    return display


def _get_chain():
    """懒加载 RAG 链（首次调用时构建，之后复用同一实例）。"""
    global _chain
    if _chain is None:
        from src.chains import build_rag_chain
        _chain = build_rag_chain()
    return _chain


def _reset_chain():
    """
    重置链和对话历史。
    在以下场景调用：
      - 上传新文件并索引完成后（知识库变了，链需要重建）
      - 删除文件后
      - 用户点击"清空对话"
    """
    global _chain, _chat_history
    _chain = None
    _chat_history = []


# ── 索引功能 ──────────────────────────────────────────────────────────────────

def ingest_any(dir_path: str, uploaded_files, force: bool):
    """
    支持两种输入方式（可同时使用）：
      1. 目录路径：填入本地路径，递归扫描所有支持格式的文件
      2. 上传文件：通过浏览器上传 PDF / Markdown / TXT

    使用 yield 实现实时进度更新（Gradio generator 模式）：
    每处理完一批文件就 yield 一次状态，前端立即刷新显示。
    普通 return 只能在函数结束时返回一次，用户看不到中间过程。

    增量索引逻辑：
      对比文件当前 MD5 和注册表中记录的 MD5，只处理有变化的文件。
      force=True 时跳过对比，重新处理所有文件。
    """
    from src.registry import (
        load_registry, get_all_doc_files,
        register_files, save_registry, SUPPORTED_EXTS,
    )
    from src.loader import load_files, split_documents
    from src.vectorstore import add_documents

    candidate_files: list[Path] = []  # 本次扫描到的所有候选文件
    errors: list[str] = []            # 路径错误或不支持的文件类型提示

    # ── 处理目录/文件路径输入 ──
    dir_path = (dir_path or "").strip()
    if dir_path:
        target = Path(dir_path).expanduser().resolve()  # 展开 ~ 并转为绝对路径
        if not target.exists():
            errors.append(f"路径不存在：{target}")
        elif target.is_file():
            # 输入的是单个文件路径
            if target.suffix.lower() in SUPPORTED_EXTS:
                candidate_files.append(target)
            else:
                errors.append(f"不支持的文件类型：{target.name}")
        elif target.is_dir():
            # 输入的是目录，递归扫描所有支持格式的文件
            candidate_files.extend(get_all_doc_files(target))

    # ── 处理浏览器上传的文件 ──
    for f in (uploaded_files or []):
        p = Path(f.name)  # Gradio 把上传文件保存到临时路径，f.name 是该路径
        if p.suffix.lower() in SUPPORTED_EXTS:
            candidate_files.append(p)
        else:
            errors.append(f"跳过不支持的文件：{p.name}")

    # 没有找到任何文件
    if not candidate_files and not errors:
        yield "请输入目录路径或上传文件。", _get_doc_table()
        return
    if not candidate_files:
        yield "\n".join(errors), _get_doc_table()
        return

    # ── 增量过滤：只保留新增或内容有变化的文件 ──
    registry = {} if force else load_registry()
    new_files = [
        p for p in candidate_files
        if force or registry.get(str(p)) != _md5(p)
    ]

    # 错误信息前缀（如果有不支持的文件，在进度里一并显示）
    prefix = ("\n".join(errors) + "\n") if errors else ""

    if not new_files:
        yield prefix + f"✓ 共 {len(candidate_files)} 个文件，全部已是最新，无需重新索引。", _get_doc_table()
        return

    total = len(new_files)
    yield prefix + f"共 {total} 个文件待处理，开始索引...", _get_doc_table()

    from src.vectorstore import get_vectorstore, _CHROMA_BATCH

    total_chunks = 0
    for i, file in enumerate(new_files, 1):
        # 阶段 1：加载文件内容
        yield prefix + f"[{i}/{total}] 解析中：{file.name}", _get_doc_table()
        docs = load_files([file])
        if not docs:
            yield prefix + f"[{i}/{total}] ⚠ 跳过（解析失败）：{file.name}", _get_doc_table()
            continue

        # 阶段 2：切块
        chunks = split_documents(docs)
        if not chunks:
            yield prefix + f"[{i}/{total}] ⚠ 跳过（内容为空）：{file.name}", _get_doc_table()
            continue

        n = len(chunks)
        n_batches = (n + _CHROMA_BATCH - 1) // _CHROMA_BATCH
        vs = get_vectorstore()

        # 阶段 3：分批向量化并写入 ChromaDB
        # 每批 500 条，写完一批就 yield 一次进度，用户看到进度在动而不是假死
        for b, start in enumerate(range(0, n, _CHROMA_BATCH), 1):
            batch = chunks[start : start + _CHROMA_BATCH]
            vs.add_documents(batch)  # 内部：Embedding 模型将文本转为向量 → upsert 到 ChromaDB
            yield (
                prefix + f"[{i}/{total}] 向量化 {file.name}："
                f"{min(start + _CHROMA_BATCH, n)}/{n} 块"
                + (f"（第 {b}/{n_batches} 批）" if n_batches > 1 else ""),
                _get_doc_table(),
            )

        # 每处理完一个文件立即更新注册表（防止中途崩溃丢失进度）
        register_files([file], registry)
        save_registry(registry)
        total_chunks += n

    # 索引完成后重置链，确保下次问答使用最新的向量库
    _reset_chain()
    yield prefix + f"✓ 全部完成：{total} 个文件，{total_chunks} 个文本块", _get_doc_table()


# ── 文档列表 ──────────────────────────────────────────────────────────────────

def _get_doc_table() -> str:
    """
    生成已索引文件的展示文本。
    数据来自注册表（JSON 文件），不查询 ChromaDB，速度快且不会触发 SQL 变量超限。
    """
    from src.vectorstore import list_sources
    sources = list_sources()
    if not sources:
        return "（暂无已索引文件）"
    lines = [f"[{i+1}] {s['name']}\n    {s['source']}" for i, s in enumerate(sources)]
    return "\n\n".join(lines)


def refresh_docs():
    """刷新按钮的回调：重新读取注册表并更新文档列表显示。"""
    return _get_doc_table()


def delete_doc(selected_name: str):
    """
    删除指定文件的索引：
      1. 从 ChromaDB 删除该文件的所有向量块
      2. 从注册表移除记录
      3. 重置 RAG 链（知识库变了，需要重建）
    """
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


# ── 聊天功能 ──────────────────────────────────────────────────────────────────

def chat(user_message: str, history: list):
    """
    聊天回调函数，由 Gradio 的发送按钮和 Enter 键触发。

    参数 history 是 Gradio Chatbot 组件传来的当前显示内容，
    但我们不直接使用它（格式在不同 Gradio 版本间不一致），
    而是用全局 _chat_history 维护 LangChain 消息历史，
    用 _build_display() 重建 Gradio 需要的显示格式。
    """
    global _chat_history

    if not user_message.strip():
        return _build_display(), ""

    from src.chains import ask_with_history
    try:
        # 调用 RAG 链：检索 + 生成
        # ask_with_history 会把本轮对话追加进 _chat_history 并返回更新后的版本
        answer, _chat_history, sources = ask_with_history(
            user_message, _chat_history, _get_chain()
        )
    except Exception as e:
        # Ollama 未启动、知识库为空等异常情况的友好提示
        answer = f"[错误] {e}\n\n请确认 Ollama 服务已启动，并运行过 `python app.py ingest`。"
        _chat_history = _chat_history + [
            HumanMessage(content=user_message),
            AIMessage(content=answer),
        ]
        sources = []

    # 如果有来源，把来源信息拼接到答案末尾（仅用于显示，不写入 _chat_history）
    if sources:
        src_lines = "\n".join(
            f"- {s['file']} 第{s['page']}页：{s['snippet']}..."
            for s in sources
        )
        answer += f"\n\n---\n**参考来源**\n{src_lines}"

    # 重建显示列表，并用带来源的 answer 覆盖最后一条助手消息
    return _build_display(override_last=answer), ""


def clear_history():
    """清空对话历史，重置 RAG 链。"""
    global _chat_history
    _chat_history = []
    return [], ""


# ── UI 构建 ───────────────────────────────────────────────────────────────────

def build_ui():
    """
    用 gr.Blocks 搭建自定义布局（比 gr.ChatInterface 更灵活）：
      左侧列：文档管理（索引、查看、删除）
      右侧列：聊天界面

    事件绑定说明：
      .click()  → 点击按钮触发
      .submit() → 文本框按 Enter 触发
      inputs    → 从哪些组件读取输入值
      outputs   → 把返回值写回哪些组件
    """
    with gr.Blocks(title="个人知识库问答") as demo:
        gr.Markdown("# 个人知识库问答系统\n基于 LangChain + ChromaDB + Ollama (gemma4:31b)")

        with gr.Row():
            # ── 左侧：文档管理面板 ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## 文档管理")

                # 输入方式 1：目录或文件的本地路径
                dir_input = gr.Textbox(
                    label="目录或文件路径（支持 ~ 展开）",
                    placeholder="/Users/jungang/Documents/课程笔记",
                    lines=1,
                )
                # 输入方式 2：直接上传文件
                file_upload = gr.File(
                    label="或直接上传文件（PDF / Markdown / TXT，可多选）",
                    file_types=[".pdf", ".md", ".txt", ".rst"],
                    file_count="multiple",
                )
                # 是否强制重建（忽略 MD5 缓存）
                force_checkbox = gr.Checkbox(label="强制重建（忽略缓存，重新索引所有文件）", value=False)
                ingest_btn = gr.Button("建立索引", variant="primary")
                # 进度显示框：ingest_any 是 generator，每次 yield 都会更新这里
                ingest_status = gr.Textbox(label="进度", interactive=False, lines=3)

                gr.Markdown("### 已索引文件")
                # 用 Textbox 而不是 Dataframe，文本内容可以直接选中复制
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

            # ── 右侧：聊天面板 ────────────────────────────────────────────────
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

        # 建立索引（generator 函数，自动流式更新进度框）
        ingest_btn.click(
            ingest_any,
            inputs=[dir_input, file_upload, force_checkbox],
            outputs=[ingest_status, doc_table],
        )

        # 刷新文档列表
        refresh_btn.click(refresh_docs, outputs=[doc_table])

        # 删除文件索引
        delete_btn.click(
            delete_doc,
            inputs=[delete_input],
            outputs=[delete_status, doc_table],
        )

        # 发送按钮 和 Enter 键都触发同一个 chat 函数
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

        # 清空对话历史
        clear_btn.click(clear_history, outputs=[chatbot, msg_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # server_name="0.0.0.0" 允许局域网内其他设备访问
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())

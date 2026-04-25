#!/usr/bin/env python3
"""
个人知识库问答系统 — 命令行入口
用法：
  python app.py ingest          # 增量建立索引（只处理新/变更文件）
  python app.py ingest --full   # 强制重建全部索引
  python app.py ask "第2章讲了什么"
  python app.py chat            # 多轮对话（带历史记忆）
  python app.py list            # 列出已索引文件
  python app.py delete <文件名>  # 删除某文件的索引
"""
import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


# ── ingest 命令 ───────────────────────────────────────────────────────────────

def cmd_ingest(args):
    """
    扫描 docs/ 目录，对新增或变更的文件建立向量索引。
    --full 参数：清空注册表，强制重新索引所有文件。
    """
    from src.registry import load_registry, get_new_files, register_files, save_registry
    from src.loader import load_files, split_documents
    from src.vectorstore import add_documents
    from src.config import DOCS_DIR

    # --full 时传空字典，让所有文件都被视为"新文件"
    registry = {} if args.full else load_registry()

    new_files = get_new_files(DOCS_DIR, registry)
    if not new_files:
        console.print("[green]✓ 没有新文件，索引已是最新。[/green]")
        return

    console.print(f"[blue]发现 {len(new_files)} 个新/变更文件，开始处理...[/blue]")
    for f in new_files:
        console.print(f"  • {f.name}")

    # 加载文档 → 切块
    docs = load_files(new_files)
    if not docs:
        console.print("[yellow]没有可解析的内容。[/yellow]")
        return

    chunks = split_documents(docs)
    console.print(f"[blue]切分为 {len(chunks)} 个文本块，正在向量化...[/blue]")

    # 向量化并写入 ChromaDB
    add_documents(chunks)

    # 更新注册表（记录已处理文件的 MD5，下次跳过）
    register_files(new_files, registry)
    save_registry(registry)

    console.print(f"[bold green]✓ 索引完成！已处理 {len(new_files)} 个文件，{len(chunks)} 个文本块。[/bold green]")


# ── ask 命令 ──────────────────────────────────────────────────────────────────

def cmd_ask(args):
    """
    单次问答：提问 → 检索 → 生成答案 → 打印。
    无对话历史，每次独立处理。
    """
    from src.chains import ask_once
    console.print(f"[cyan]问题：{args.question}[/cyan]\n")
    result = ask_once(args.question)
    # Panel + Markdown 渲染：让答案支持标题、加粗、列表等格式
    console.print(Panel(Markdown(result["answer"]), title="[bold green]答案[/bold green]"))
    _print_sources(result["sources"])


# ── chat 命令 ─────────────────────────────────────────────────────────────────

def cmd_chat(args):
    """
    交互式多轮对话：
    - 共享同一个 RAG 链实例，避免重复初始化
    - history 列表在整个会话中持续累积，实现上下文记忆
    - 输入 exit / quit / q 退出
    """
    from src.chains import build_rag_chain, ask_with_history
    console.print(Panel(
        "多轮对话模式（带历史记忆）\n输入问题后按 Enter，输入 [bold]exit[/bold] 退出",
        style="blue",
    ))
    chain = build_rag_chain()
    history = []  # 存储 HumanMessage / AIMessage 交替的历史记录
    while True:
        try:
            q = console.input("\n[bold cyan]你 > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        # ask_with_history 会把本轮对话追加进 history 并返回
        answer, history, sources = ask_with_history(q, history, chain)
        console.print(Panel(Markdown(answer), title="[bold green]助手[/bold green]"))
        _print_sources(sources)
    console.print("[dim]再见！[/dim]")


# ── list 命令 ─────────────────────────────────────────────────────────────────

def cmd_list(args):
    """
    列出所有已索引文件，数据直接读注册表，不查询 ChromaDB。
    """
    from src.vectorstore import list_sources
    sources = list_sources()
    if not sources:
        console.print("[yellow]知识库为空，请先运行 ingest。[/yellow]")
        return
    t = Table(title="已索引文件", show_lines=True)
    t.add_column("#", style="dim", width=4)
    t.add_column("文件名", style="cyan")
    t.add_column("完整路径", style="dim")
    for i, s in enumerate(sources, 1):
        t.add_row(str(i), s["name"], s["source"])
    console.print(t)


# ── delete 命令 ───────────────────────────────────────────────────────────────

def cmd_delete(args):
    """
    删除指定文件的所有向量块，并从注册表中移除记录。
    支持文件名（lecture1.pdf）或完整路径两种输入方式。
    """
    from src.vectorstore import delete_by_source, list_sources
    from src.registry import load_registry, unregister_file, save_registry

    filename = args.filename
    sources = list_sources()
    # 在已索引列表中找到匹配项（文件名或完整路径均可）
    match = next((s for s in sources if s["name"] == filename or s["source"] == filename), None)

    if not match:
        console.print(f"[yellow]未找到文件 '{filename}'，使用 list 查看已索引文件。[/yellow]")
        return

    # 从 ChromaDB 删除该文件的所有向量块
    count = delete_by_source(match["source"])

    # 从注册表移除记录（下次 ingest 时会重新处理该文件）
    registry = load_registry()
    unregister_file(match["source"], registry)
    save_registry(registry)

    console.print(f"[green]✓ 已删除 '{match['name']}' 的 {count} 个文本块。[/green]")


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _print_sources(sources: list):
    """打印参考来源列表，供 ask 和 chat 命令复用。"""
    if not sources:
        return
    console.print("\n[dim]── 参考来源 ──[/dim]")
    for i, s in enumerate(sources, 1):
        console.print(
            f"  [{i}] [yellow]{s['file']}[/yellow] 第 {s['page']} 页\n"
            f"      {s['snippet']}..."
        )


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="个人知识库问答系统")
    sub = parser.add_subparsers(dest="cmd")

    # ingest 子命令
    p_ingest = sub.add_parser("ingest", help="建立/更新向量索引")
    p_ingest.add_argument("--full", action="store_true", help="忽略缓存，重建全部索引")

    # ask 子命令
    p_ask = sub.add_parser("ask", help="单次提问")
    p_ask.add_argument("question", help="你的问题")

    # 其他子命令（无额外参数）
    sub.add_parser("chat", help="多轮对话模式")
    sub.add_parser("list", help="列出已索引文件")

    # delete 子命令
    p_del = sub.add_parser("delete", help="删除某文件的索引")
    p_del.add_argument("filename", help="文件名（如 lecture1.pdf）")

    args = parser.parse_args()

    # 用字典分发，避免一堆 if-elif
    dispatch = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "list": cmd_list,
        "delete": cmd_delete,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

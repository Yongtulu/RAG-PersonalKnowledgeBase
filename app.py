#!/usr/bin/env python3
"""
个人知识库问答系统 CLI
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


# ── ingest ────────────────────────────────────────────────────────────────────

def cmd_ingest(args):
    from src.registry import load_registry, get_new_files, register_files, save_registry
    from src.loader import load_files, split_documents
    from src.vectorstore import add_documents
    from src.config import DOCS_DIR

    registry = {} if args.full else load_registry()

    new_files = get_new_files(DOCS_DIR, registry)
    if not new_files:
        console.print("[green]✓ 没有新文件，索引已是最新。[/green]")
        return

    console.print(f"[blue]发现 {len(new_files)} 个新/变更文件，开始处理...[/blue]")
    for f in new_files:
        console.print(f"  • {f.name}")

    docs = load_files(new_files)
    if not docs:
        console.print("[yellow]没有可解析的内容。[/yellow]")
        return

    chunks = split_documents(docs)
    console.print(f"[blue]切分为 {len(chunks)} 个文本块，正在向量化...[/blue]")

    add_documents(chunks)
    register_files(new_files, registry)
    save_registry(registry)

    console.print(f"[bold green]✓ 索引完成！已处理 {len(new_files)} 个文件，{len(chunks)} 个文本块。[/bold green]")


# ── ask ───────────────────────────────────────────────────────────────────────

def cmd_ask(args):
    from src.chains import ask_once
    console.print(f"[cyan]问题：{args.question}[/cyan]\n")
    result = ask_once(args.question)
    console.print(Panel(Markdown(result["answer"]), title="[bold green]答案[/bold green]"))
    _print_sources(result["sources"])


# ── chat ──────────────────────────────────────────────────────────────────────

def cmd_chat(args):
    from src.chains import build_rag_chain, ask_with_history
    console.print(Panel(
        "多轮对话模式（带历史记忆）\n输入问题后按 Enter，输入 [bold]exit[/bold] 退出",
        style="blue",
    ))
    chain = build_rag_chain()
    history = []
    while True:
        try:
            q = console.input("\n[bold cyan]你 > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        answer, history, sources = ask_with_history(q, history, chain)
        console.print(Panel(Markdown(answer), title="[bold green]助手[/bold green]"))
        _print_sources(sources)
    console.print("[dim]再见！[/dim]")


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list(args):
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


# ── delete ────────────────────────────────────────────────────────────────────

def cmd_delete(args):
    from src.vectorstore import delete_by_source, list_sources
    from src.registry import load_registry, unregister_file, save_registry

    filename = args.filename
    sources = list_sources()
    match = next((s for s in sources if s["name"] == filename or s["source"] == filename), None)

    if not match:
        console.print(f"[yellow]未找到文件 '{filename}'，使用 list 查看已索引文件。[/yellow]")
        return

    count = delete_by_source(match["source"])
    registry = load_registry()
    unregister_file(match["source"], registry)
    save_registry(registry)

    console.print(f"[green]✓ 已删除 '{match['name']}' 的 {count} 个文本块。[/green]")


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_sources(sources: list):
    if not sources:
        return
    console.print("\n[dim]── 参考来源 ──[/dim]")
    for i, s in enumerate(sources, 1):
        console.print(
            f"  [{i}] [yellow]{s['file']}[/yellow] 第 {s['page']} 页\n"
            f"      {s['snippet']}..."
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="个人知识库问答系统")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="建立/更新向量索引")
    p_ingest.add_argument("--full", action="store_true", help="忽略缓存，重建全部索引")

    p_ask = sub.add_parser("ask", help="单次提问")
    p_ask.add_argument("question", help="你的问题")

    sub.add_parser("chat", help="多轮对话模式")
    sub.add_parser("list", help="列出已索引文件")

    p_del = sub.add_parser("delete", help="删除某文件的索引")
    p_del.add_argument("filename", help="文件名（如 lecture1.pdf）")

    args = parser.parse_args()
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

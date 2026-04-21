# Personal Knowledge Base Q&A

A fully local RAG (Retrieval-Augmented Generation) system built with **LangChain + ChromaDB + Ollama**.  
Chat with your own documents — course notes, PDFs, Markdown files — entirely offline.

## Features

- **Incremental indexing** — MD5-based change detection; only new or modified files are re-processed
- **Multi-turn memory** — conversation history is preserved across turns using LangChain LCEL
- **Web UI** — Gradio interface with directory/file input, real-time indexing progress, and document management
- **CLI** — `ask`, `chat`, `list`, `delete` commands for terminal use
- **Multilingual** — embedding model supports Chinese, English, and mixed-language documents

## Tech Stack

| Component | Choice |
|-----------|--------|
| RAG Framework | LangChain (LCEL) |
| Vector Store | ChromaDB (local, persistent) |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (~118 MB, runs locally) |
| LLM | Ollama + `gemma4:31b` (fully offline) |
| Web UI | Gradio |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```bash
# Install Ollama: https://ollama.ai
ollama pull gemma4:31b
```

### 3. Configure (optional)

```bash
cp .env.example .env
# Edit .env to change the model or retrieval parameters
```

### 4. Add your documents

Place your notes, PDFs, or text files in the `docs/` directory (subdirectories supported):

```
docs/
├── machine-learning/
│   ├── chapter1.pdf
│   └── notes.md
└── computer-networks.txt
```

### 5. Build the index

```bash
python app.py ingest            # incremental — only processes new/changed files
python app.py ingest --full     # force rebuild from scratch
```

### 6. Ask questions

```bash
# Single question
python app.py ask "What does chapter 2 cover?"

# Multi-turn conversation (with memory)
python app.py chat

# Launch the Web UI (visit http://localhost:7860)
python webui.py
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `python app.py ingest` | Incrementally index new/changed files |
| `python app.py ingest --full` | Rebuild entire index from scratch |
| `python app.py ask "<question>"` | One-shot Q&A |
| `python app.py chat` | Interactive multi-turn chat |
| `python app.py list` | List all indexed files |
| `python app.py delete <filename>` | Remove a file's index entries |

## Web UI

Run `python webui.py` and open [http://localhost:7860](http://localhost:7860).

**Left panel — Document Management**
- Enter a directory path or upload files directly (PDF / Markdown / TXT)
- Real-time progress: file count, chunk count, per-batch vectorization status
- View and delete indexed documents

**Right panel — Chat**
- Ask questions and get answers with cited sources
- Conversation history persists across turns; click "Clear" to reset

## Project Structure

```
├── app.py              # CLI entry point
├── webui.py            # Gradio Web UI
├── src/
│   ├── config.py       # Global configuration (paths, model names, parameters)
│   ├── registry.py     # Incremental index — file hash registry
│   ├── loader.py       # Document loading and text splitting
│   ├── vectorstore.py  # ChromaDB operations (add, delete, retrieve)
│   ├── chains.py       # LangChain RAG chain with multi-turn memory
│   └── prompts.py      # Prompt templates (QA + condense)
├── docs/               # Your documents go here (git-ignored)
├── chroma_db/          # Vector index — auto-generated (git-ignored)
├── .env.example        # Environment variable template
└── requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```env
OLLAMA_MODEL=gemma4:31b          # any model available in your Ollama
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=paraphrase-multilingual-MiniLM-L12-v2
TOP_K=5                          # number of chunks retrieved per query
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with your chosen model pulled
- ~120 MB disk space for the embedding model (downloaded automatically on first run)

## License

MIT

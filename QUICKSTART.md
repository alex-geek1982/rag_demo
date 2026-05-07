# RAG Engine — Quick Start

A 5-minute path from an empty checkout to a working hybrid PDF RAG pipeline.

## 1. Prerequisites

- Python 3.10+
- An API key for one of: OpenAI, DeepBricks, Azure OpenAI, or Google Gemini
- (Optional) [Ollama](https://ollama.ai) for local embedding/reranking

## 2. Install

```bash
pip install -e .
# or, with dev tools
pip install -e ".[dev]"
```

## 3. Configure

```bash
cp .env.example .env
# Edit .env with one of the provider configs below
```

### OpenAI / DeepBricks (default)
```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Azure OpenAI (one-line switch)
```bash
USE_AZURE_OPENAI=true
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_VISION_DEPLOYMENT=gpt-4o
```

### Gemini Vision (PDF image analysis only)
```bash
VISION_PROVIDER=gemini
GEMINI_API_KEY=AIza...
```

## 4. Run the Featured Example

```bash
python examples/hybrid_pdf_rag_chroma_kuzu.py
```

This processes a sample PDF end-to-end:
1. Parse the PDF (with optional Vision-based image understanding)
2. Chunk and embed
3. Index into **Chroma** (vector + BM25)
4. Extract entities & relationships with the LLM
5. Embed entities (back into Chroma) and write the graph to **Kuzu**
6. Run a hybrid query (vector + BM25 + graph) → rerank → answer

### Toggle individual stages

The example respects four boolean env flags:

```bash
# Re-query an already-built DB without reprocessing anything
UPDATE_KB=false UPDATE_KG=false BUILD_BM25=false EXECUTE_QUERY=true \
  python examples/hybrid_pdf_rag_chroma_kuzu.py

# Iterate on the KG extraction logic without re-parsing the PDF
UPDATE_KB=false UPDATE_KG=true BUILD_BM25=false EXECUTE_QUERY=false \
  python examples/hybrid_pdf_rag_chroma_kuzu.py

# Just (re)build the BM25 index
UPDATE_KB=false UPDATE_KG=false BUILD_BM25=true EXECUTE_QUERY=false \
  python examples/hybrid_pdf_rag_chroma_kuzu.py
```

| Flag | What it does |
|------|--------------|
| `UPDATE_KB` | Parse PDF → embeddings → Chroma + KG |
| `UPDATE_KG` | Stateless KG rebuild from existing Chroma chunks |
| `BUILD_BM25` | Build BM25 index from Chroma chunks |
| `EXECUTE_QUERY` | Hybrid retrieval → rerank → answer |

## 5. Run Other Examples

```bash
python examples/basic_example.py                  # Minimal end-to-end RAG
python examples/bm25_demo.py                      # BM25 standalone
python examples/hybrid_retrieval_example.py       # Vector + BM25 fusion
python examples/multilingual_rag_example.py       # Cross-lingual retrieval
python examples/gemini_pdf_rag_example.py         # Gemini Vision for PDFs
python examples/pdf_processing_example.py         # Advanced PDF parser
python examples/advanced_chunking_example.py      # TitleChunker hierarchy
python examples/lightrag_kg_example.py            # LightRAG-style KG
python examples/multimodal_example.py             # Mixed text + images + tables
```

## 6. Programmatic Use

### Modular pipeline (preferred)
```python
from rag_engine.config import RAGEngineConfig
from rag_engine.pipeline import (
    DocumentProcessor, KnowledgeBaseBuilder, KnowledgeGraphBuilder,
)
from rag_engine.storage import ChromaKnowledgeBase, KuzuGraphStore

config = RAGEngineConfig.from_env()

doc = DocumentProcessor(config).process_document("paper.pdf", language="en")

kb = KnowledgeBaseBuilder(config)
kb.build_from_document(doc)
chroma = ChromaKnowledgeBase("./output/chroma_db")
kb.rebuild_chroma(chroma)

kg = KnowledgeGraphBuilder(config)
blocks = kg.merge_chunks_by_token_size(doc.chunks)
entities, rels = kg.extract_entities_and_relationships(blocks)
kg.embed_entities_and_store_to_chroma(entities, chroma)

kuzu = KuzuGraphStore("./output/kuzu_db")
kg.store_entities_and_relationships_to_kuzu(kuzu, {b.id: b for b in blocks})

chroma.build_bm25_index_from_chroma()
```

### Querying
```python
import asyncio
from rag_engine.pipeline.retrieval_pipeline import (
    RetrievalPipeline, LocalReranker, LocalAnswerGenerator,
)

pipeline = RetrievalPipeline(config)
reranker = LocalReranker("http://localhost:11434", "qllama/bge-reranker-v2-m3:q4_k_m")
generator = LocalAnswerGenerator(
    base_url=config.llm.base_url,
    api_key=config.llm.api_key,
    model=config.llm.model,
)

result = asyncio.run(pipeline.run_query(
    query="What is the main contribution?",
    chroma_kb=chroma,
    kuzu_store=kuzu,
    reranker=reranker,
    generator=generator,
    top_k=5,
))
print(result["answer"])
```

### High-level API
```python
from rag_engine.core import RAGEngine
from rag_engine.config import RAGEngineConfig

engine = RAGEngine(RAGEngineConfig.from_env())
engine.process_document("paper.pdf")
print(engine.query("What is the main contribution?").answer)
```

## 7. Common Tweaks

| Goal | Env var | Value |
|------|---------|-------|
| Use token-based chunking | `CHUNKER_TYPE` | `token` (default `title`) |
| Bigger chunks | `CHUNK_SIZE` / `CHUNK_OVERLAP` | e.g. `2048` / `400` |
| Disable BM25 | `ENABLE_BM25` | `false` |
| Switch fusion strategy | `FUSION_STRATEGY` | `rrf`, `weighted_avg`, `max`, `min` |
| Adjust channel weights | `VECTOR_WEIGHT` / `BM25_WEIGHT` / `GRAPH_WEIGHT` | sum to 1.0 (auto-normalized otherwise) |
| Cross-encoder reranker | `RERANK_MODEL` | `cross-encoder` |
| Stronger reranker | `RERANK_MODEL` | `llm` or `hybrid` |
| Skip header/footer in PDFs | `PDF_FILTER_HEADER_FOOTER` | `true` (default) |
| Default language | `LANGUAGE` | `en`, `zh`, `ja`, `ko`, `es`, `fr`, `de` |

See [README.md](README.md) for the full configuration reference.

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'openai'`
```bash
pip install -e .
```

### `AuthenticationError: Invalid API key`
Confirm the right env var is set for the provider you're using:
```bash
grep -E "OPENAI_API_KEY|AZURE_OPENAI_API_KEY|GEMINI_API_KEY|DEEPBRICKS_API_KEY" .env
```

### `Chroma DB exists, but no data`
You ran a workflow that depends on existing data without first running `UPDATE_KB=true`. Run the full pipeline once, then disable `UPDATE_KB` to iterate.

### Kuzu DB lock or schema errors
Delete `output/<demo>/kuzu_db` and re-run with `UPDATE_KG=true` (or `UPDATE_KB=true`).

### `ValueError: No parser found for file`
Supported formats are PDF, DOCX, XLSX, JPG/PNG/GIF, TXT, MD. To add a new format, subclass `BaseParser` and register it with `ParserFactory` (see [ARCHITECTURE.md](ARCHITECTURE.md#extension-points)).

## 9. Next Steps

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for the layered architecture and data flows.
2. Skim [README.md](README.md) for the full feature & config reference.
3. Try `examples/hybrid_pdf_rag_chroma_kuzu.py` against your own PDF (set `RAG_PDF_PATH=/path/to/file.pdf`).
4. Iterate on KG extraction with `UPDATE_KB=false UPDATE_KG=true` — fast feedback loop without re-parsing PDFs.

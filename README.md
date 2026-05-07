# RAG Engine — Modular Multimodal RAG Framework

> A modular, decoupled RAG engine with hybrid retrieval (vector + BM25 + graph), advanced PDF parsing, multilingual support, and pluggable storage backends (Chroma + Kuzu).

## Core Features

### Modular, Decoupled Architecture
The engine is organized as independent layers that can be composed or used standalone:

- **Pipeline layer** — `DocumentProcessor`, `KnowledgeBaseBuilder`, `KnowledgeGraphBuilder`, `RetrievalPipeline`
- **Storage layer** — `ChromaKnowledgeBase` (vectors + BM25 + entity index) and `KuzuGraphStore` (LPG graph DB), both fully self-contained
- **Retrieval layer** — Hybrid fusion (vector + BM25 + graph), multiple normalization and rerank strategies
- **Parsers / Processors** — Pluggable per-format and per-modality components

You can run any layer on its own (rebuild a graph from existing Chroma data, build a BM25 index without touching the LLM, etc.).

### Hybrid Retrieval
- **Vector search** via Chroma (HNSW)
- **BM25 full-text search** with configurable `k1`, `b`, language, min token length
- **Graph search** via Kuzu (n-hop entity traversal, PageRank scoring)
- **Fusion strategies**: weighted average, RRF (Reciprocal Rank Fusion), max, min
- **Score normalization**: minmax, sigmoid, rank
- **Deduplication** across channels

### Multimodal Document Processing
- **Advanced PDF parser** with layout analysis, image extraction, table detection, header/footer filtering
- **Vision-based image understanding** with OpenAI, Azure OpenAI, or Google Gemini
- **Native handlers** for text, images, tables, equations, code
- **Format support**: PDF, DOCX, XLSX, JPG/PNG/GIF, TXT, MD

### Knowledge Graph
- **LLM-based entity & relationship extraction** (LightRAG-style multi-turn prompting)
- **Entity embeddings** stored alongside chunks in Chroma
- **Persistent graph** in Kuzu with schema-managed nodes and edges
- **Two rebuild paths**: from a parsed document, or directly from existing Chroma chunks

### Reranking
Pluggable rerankers with shared interface:
- `SimpleReranker` — heuristic
- `CrossEncoderReranker` — `cross-encoder/ms-marco-MiniLM-L-12-v2` by default
- `LLMReranker` — LLM-based scoring (Ollama or OpenAI-compatible)
- `HybridReranker` — combines multiple strategies

### Multilingual Support (7 languages)
English, Chinese, Japanese, Korean, Spanish, French, German.
- Automatic language detection via Unicode-range heuristics
- Multilingual embeddings with language-aware tagging
- Cross-lingual retrieval (query in one language, retrieve in others)

### Azure OpenAI
First-class Azure OpenAI support across embedding, LLM, and vision components, with a single `USE_AZURE_OPENAI` master switch and per-component overrides.

## Quick Start

### Install
```bash
pip install -e .
# or
pip install -e ".[dev]"
```

### Configure
```bash
cp .env.example .env
# Edit .env to set OPENAI_API_KEY (or DEEPBRICKS_API_KEY / AZURE_OPENAI_* / GEMINI_API_KEY)
```

### Run the featured example
```bash
python examples/hybrid_pdf_rag_chroma_kuzu.py
```

This runs the full pipeline against a sample PDF: parse → chunk → embed → index in Chroma → extract entities/relations with the LLM → embed entities → write to Kuzu → build BM25 index → run a hybrid query.

## Featured Example: hybrid_pdf_rag_chroma_kuzu.py

The example exposes four independent workflow flags so you can run any subset:

| Flag | Default | What it does |
|------|---------|--------------|
| `UPDATE_KB` | `true` | Parse the PDF, build embeddings, rebuild Chroma, then extract entities & write to Kuzu |
| `UPDATE_KG` | `true` | **Stateless graph rebuild from existing Chroma chunks** — no PDF reprocessing |
| `BUILD_BM25` | `true` | Read Chroma chunks and build a persistent BM25 index (`bm25_index.pkl`) |
| `EXECUTE_QUERY` | `true` | Run hybrid retrieval (vector + BM25 + graph) → rerank → generate answer |

Set any to `false` (e.g. `UPDATE_KB=false UPDATE_KG=false BUILD_BM25=false EXECUTE_QUERY=true`) to query an already-built database.

## Programmatic Usage

### Modular pipeline (recommended)
```python
from rag_engine.config import RAGEngineConfig
from rag_engine.pipeline import (
    DocumentProcessor,
    KnowledgeBaseBuilder,
    KnowledgeGraphBuilder,
    RetrievalPipeline,
)
from rag_engine.storage import ChromaKnowledgeBase, KuzuGraphStore

config = RAGEngineConfig.from_env()

# 1. Parse the document
doc_processor = DocumentProcessor(config)
document = doc_processor.process_document("report.pdf", language="en")

# 2. Build vector KB
kb_builder = KnowledgeBaseBuilder(config)
kb_builder.build_from_document(document)

chroma_kb = ChromaKnowledgeBase("./output/chroma_db")
kb_builder.rebuild_chroma(chroma_kb)

# 3. Build knowledge graph
kg_builder = KnowledgeGraphBuilder(config)
content_blocks = kg_builder.merge_chunks_by_token_size(document.chunks)
entities, relationships = kg_builder.extract_entities_and_relationships(content_blocks)
kg_builder.embed_entities_and_store_to_chroma(entities, chroma_kb)

kuzu_store = KuzuGraphStore("./output/kuzu_db")
kg_builder.store_entities_and_relationships_to_kuzu(
    kuzu_store, {b.id: b for b in content_blocks}
)

# 4. Build BM25
chroma_kb.build_bm25_index_from_chroma()
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
    query="What are the key parameters?",
    chroma_kb=chroma_kb,
    kuzu_store=kuzu_store,
    reranker=reranker,
    generator=generator,
    top_k=5,
))
print(result["answer"])
```

### High-level RAGEngine API (still supported)
```python
from rag_engine.core import RAGEngine
from rag_engine.config import RAGEngineConfig

engine = RAGEngine(RAGEngineConfig.from_env())
engine.process_document("report.pdf")
result = engine.query("What is the main topic?", top_k=5)
print(result.answer)
```

## Configuration

`RAGEngineConfig` is composed of typed dataclasses. All fields have environment-variable defaults.

| Sub-config | Purpose | Key fields |
|------------|---------|-----------|
| `EmbeddingConfig` | Embedding provider | `model`, `dimension`, `api_key`, `base_url`, `use_azure`, `azure_*` |
| `LLMConfig` | LLM for extraction & answering | `model`, `temperature`, `max_tokens`, `enable_cache`, `use_azure` |
| `VisionConfig` | Vision model for PDF/image analysis | `model`, `provider` (`openai` / `azure` / `gemini`) |
| `LanguageConfig` | Multilingual settings | `default_language`, `supported_languages` |
| `PDFProcessingConfig` | Advanced PDF parsing | `use_advanced_layout`, `extract_images/tables/text`, `filter_header_footer`, `min_image_area` |
| `ProcessingConfig` | Chunking & concurrency | `chunker_type` (`title` / `token`), `chunk_size`, `chunk_overlap`, `max_workers`, `max_entity_tokens` |
| `BM25Config` | Full-text search | `enable_bm25`, `k1`, `b`, `language`, `min_token_length` |
| `HybridRetrievalConfig` | Fusion strategy | `fusion_strategy`, `vector_weight`, `bm25_weight`, `graph_weight`, `normalization_method`, `rrf_k`, `enable_dedup` |
| `RerankerConfig` | Reranker selection | `rerank_model` (`simple` / `cross-encoder` / `llm` / `hybrid`), `rerank_top_k`, `rerank_final_k` |
| `StorageConfig` | Storage backend tuning | `cosine_threshold`, `related_chunk_number`, `max_graph_nodes` |

### Environment variables (essential)
```bash
# OpenAI / DeepBricks
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI (optional master switch)
USE_AZURE_OPENAI=true
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_VISION_DEPLOYMENT=gpt-4o

# Gemini Vision (optional)
VISION_PROVIDER=gemini
GEMINI_API_KEY=AIza...

# Models
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7

# Workdirs
WORKING_DIR=./rag_storage
OUTPUT_DIR=./output

# Chunking & retrieval
CHUNKER_TYPE=title           # 'title' or 'token'
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
ENABLE_BM25=true
FUSION_STRATEGY=weighted_avg # 'weighted_avg' | 'rrf' | 'max' | 'min'
VECTOR_WEIGHT=0.5
BM25_WEIGHT=0.5
ENABLE_RERANK=true
RERANK_MODEL=simple          # 'simple' | 'cross-encoder' | 'llm' | 'hybrid'

LANGUAGE=en
```

## Other Examples

| File | Demonstrates |
|------|--------------|
| `examples/hybrid_pdf_rag_chroma_kuzu.py` | **Featured** — full modular pipeline with 4 toggleable workflows |

## Module Map

```
rag_engine/
├── core/          # RAGEngine orchestrator, KnowledgeGraph, EntityExtractor, RelationshipBuilder, prompts, llm_client
├── pipeline/      # DocumentProcessor, KnowledgeBaseBuilder, KnowledgeGraphBuilder, RetrievalPipeline, chunker
├── storage/       # ChromaKnowledgeBase, KuzuGraphStore
├── retrieval/     # OpenAIEmbedding, HybridRetriever, BM25Retriever, ScoreNormalizer, HybridFuser,
│                  # SimpleReranker / LLMReranker / CrossEncoderReranker / HybridReranker, ContextExtractor
├── parsers/       # BaseParser, PDFParser, DocxParser, ExcelParser, ImageParser, TextParser,
│                  # AdvancedPDFProcessor, ParserFactory
├── processors/    # BaseModalProcessor, TextProcessor, ImageProcessor, TableProcessor,
│                  # EquationProcessor, CodeProcessor, ProcessorFactory
├── i18n/          # I18n, LanguageDetector, MultilingualEmbedding, CrosslingualRetrieval
├── config.py      # RAGEngineConfig + sub-configs
└── types.py       # ContentType, ModalityType, ContentBlock, Document, Chunk, Entity, Relationship, RetrievalResult, QueryResult
```

## Extending

### New file format
1. Subclass `BaseParser` in `rag_engine/parsers/`
2. Implement `parse()` and `supports()`
3. Register in `ParserFactory`

### New modality
1. Subclass `BaseModalProcessor` in `rag_engine/processors/`
2. Implement `process()` and `supports()`
3. Register in `ProcessorFactory`

### New storage backend
1. Mirror the public surface of `ChromaKnowledgeBase` (`rebuild`, `search`, `count`, `get_all`) or `KuzuGraphStore`
2. Update the corresponding `KnowledgeBaseBuilder.rebuild_*()` / `KnowledgeGraphBuilder.store_*()` calls

### New retrieval / fusion strategy
Add a method to `RetrievalPipeline` or extend `HybridFuser` in `rag_engine/retrieval/`.

## Testing
```bash
pytest tests/
pytest --cov=rag_engine tests/
```

## License
MIT

# Architecture

## Layered Overview

```
┌──────────────────────────────────────────────────────────┐
│                Application & Examples                    │
│  examples/hybrid_pdf_rag_chroma_kuzu.py (featured)       │
│  examples/bm25_demo.py, multilingual_rag_example.py, ... │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                     Pipeline Layer                       │
│  ┌────────────────────────────────────────────────────┐  │
│  │ DocumentProcessor      → Document + chunks        │  │
│  │ KnowledgeBaseBuilder   → embeddings, Chroma       │  │
│  │ KnowledgeGraphBuilder  → entities, relations, KG  │  │
│  │ RetrievalPipeline      → retrieve → rerank → gen  │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                     Storage Layer                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │ ChromaKnowledgeBase  (vectors + BM25 + entities)  │  │
│  │ KuzuGraphStore       (LPG graph DB)               │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                  Retrieval & Support                     │
│  HybridRetriever | BM25Retriever | HybridFuser           │
│  Rerankers (Simple / CrossEncoder / LLM / Hybrid)        │
│  Embeddings (OpenAI / Azure / Ollama via OpenAI-compat)  │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                External Services                         │
│  Chroma DB | Kuzu DB | OpenAI | Azure OpenAI | Gemini    │
│  Ollama (local LLM/embeddings/reranker)                  │
└──────────────────────────────────────────────────────────┘
```

The two key invariants of the architecture:

1. **Storage classes are fully decoupled.** `ChromaKnowledgeBase` and `KuzuGraphStore` know nothing about the pipeline or the engine. They can be instantiated standalone and operated directly.
2. **Pipeline modules are independent.** Each can run on its own input and produce its own output. This is what makes workflows like "rebuild the KG from existing Chroma chunks without touching the PDF" possible.

## Pipeline Layer

### DocumentProcessor (`rag_engine/pipeline/document_processor.py`)
```python
DocumentProcessor
  ├── process_document(file_path, doc_id=None, doc_title=None,
  │                    language=None, markdown_path=None) → Document
  ├── process_folder(folder_path, language=None, recursive=True) → List[Document]
  └── _process_content_blocks(document)
```
**Dependencies**: `ParserFactory`, `ProcessorFactory`
**Responsibilities**: parse a file, extract content blocks, run modal processors, produce a `Document` with `chunks`.
**Properties**: pure input → output, no side effects on storage.

### KnowledgeBaseBuilder (`rag_engine/pipeline/knowledge_base_builder.py`)
```python
KnowledgeBaseBuilder
  ├── build_from_document(document)
  ├── build_from_blocks(blocks)
  ├── rebuild_chroma(chroma_kb)
  ├── get_embeddings()       → Dict[chunk_id, List[float]]
  └── get_content_blocks()   → Dict[chunk_id, ContentBlock]
```
**Dependencies**: `OpenAIEmbedding` (or Azure / Ollama via the same OpenAI-compatible interface)
**Responsibilities**: generate embeddings for chunks; coordinate a Chroma rebuild.
**Properties**: embeddings are computed in memory and held on the builder, so a Chroma rebuild can be inspected or repeated without re-embedding.

### KnowledgeGraphBuilder (`rag_engine/pipeline/knowledge_graph_builder.py`)
The KG building uses a **three-step flow**:
```python
KnowledgeGraphBuilder
  ├── merge_chunks_by_token_size(chunks, max_tokens=4096) → List[ContentBlock]
  ├── extract_entities_and_relationships(blocks)          → (entities, relationships)   # Step 1: LLM
  ├── embed_entities_and_store_to_chroma(entities, chroma_kb)                            # Step 2: vectors
  ├── store_entities_and_relationships_to_kuzu(kuzu_store, content_blocks)               # Step 3: graph DB
  ├── build_from_document(document)             # Convenience wrapper for steps 1–3
  ├── build_from_blocks(blocks)                 # Same, from prepared blocks
  ├── rebuild_kuzu(kuzu_store)                  # Reload current entities/relations into Kuzu
  ├── rebuild_kuzu_from_chroma_chunks(...)      # Stateless rebuild from Chroma data
  ├── rebuild_kuzu_from_extracted_data(kuzu_store)
  ├── rebuild_entities_chroma(chroma_kb)
  ├── get_entities() / get_relationships() / get_graph_stats()
```
**Dependencies**: `EntityExtractor`, `RelationshipBuilder`, `KnowledgeGraph` (in-memory model), `OpenAIEmbedding` for entity vectors
**Properties**: supports two rebuild paths — from a `Document` (with chunks) or from existing Chroma chunks — so the KG can be iterated without reprocessing PDFs.

### RetrievalPipeline (`rag_engine/pipeline/retrieval_pipeline.py`)
```python
RetrievalPipeline
  ├── retrieve_hybrid(query, chroma_kb, kuzu_store, top_k, ...) → List[RetrievalResult]
  └── run_query(query, chroma_kb, kuzu_store, reranker, generator, top_k) → Dict
```
Bundled helpers:
- `LocalReranker(base_url, model)` — Ollama-style reranker
- `LocalAnswerGenerator(base_url, api_key, model, ...)` — OpenAI-compatible (also supports Azure)

**Responsibilities**: vectorize the query, run Chroma + Kuzu searches in parallel, fuse and dedupe results, rerank, and synthesize an answer.

## Storage Layer

### ChromaKnowledgeBase (`rag_engine/storage/chroma_kb.py`)
```python
ChromaKnowledgeBase
  ├── __init__(db_path)
  ├── get_or_create_collection(name)
  ├── rebuild(chunks, embeddings)                    # full vector index rebuild
  ├── rebuild_entities(entities, embeddings)         # entity index for graph search
  ├── build_bm25_index_from_chroma()                 # extract docs from Chroma → BM25 → bm25_index.pkl
  ├── search(query_vector, top_k)                    → List[RetrievalResult]
  ├── search_bm25(query, top_k)                      → List[RetrievalResult]
  ├── search_entities(query_vector, top_k)           → List[RetrievalResult]
  ├── count() → int
  └── get_all(name=None) → {ids, documents, metadatas, embeddings}
```
**Properties**:
- Self-contained: holds Chroma client, BM25 index, and entity collection in one place.
- Persistent: BM25 index is pickled next to the Chroma DB.
- Multi-collection: separate spaces for chunks vs. entities.
- Exposes `get_all()` so other layers can rebuild from it without re-parsing source files.

### KuzuGraphStore (`rag_engine/storage/kuzu_graph.py`)
```python
KuzuGraphStore
  ├── __init__(db_path)
  ├── close()
  ├── __enter__() / __exit__()                       # use as context manager
  ├── rebuild_from_entities_and_relationships(entities, relationships, content_blocks)
  ├── rebuild_from_chroma_chunks(chunk_ids, documents, embeddings, entities)
  ├── rebuild_from_extracted_data(entities, relationships)
  ├── search(query_entities, top_k, n_hop)           → Dict
  ├── get_all_entities()                             → Dict[id, Entity]
  ├── _compute_pagerank()                            # entity-importance scoring
  ├── _ensure_schema() / _clear_and_recreate()
```
**Properties**:
- LPG model with managed schema for nodes (Entity, Chunk) and edges.
- Multiple rebuild paths (from extractor output, from Chroma data, from in-memory graph).
- Graph search supports n-hop traversal and PageRank-weighted scoring.

## Retrieval Layer

`rag_engine/retrieval/` provides the building blocks the pipeline uses:

- **Embeddings** — `OpenAIEmbedding` (also handles Azure & Ollama via OpenAI-compatible base URL)
- **HybridRetriever** — top-level vector + BM25 retriever
- **BM25Retriever** — pure full-text search with `rank_bm25`
- **ScoreNormalizer** — `minmax`, `sigmoid`, `rank` normalization for cross-channel score fusion
- **HybridFuser** — `weighted_avg`, `rrf`, `max`, `min` fusion strategies, with optional dedup
- **Rerankers** — `SimpleReranker`, `CrossEncoderReranker`, `LLMReranker`, `HybridReranker`
- **ContextExtractor** — pulls surrounding chunks/entities for prompting; cached via `ContextCache`

## Data Flows

### Workflow A: Full processing (`UPDATE_KB=true`)
```
PDF → DocumentProcessor → Document(chunks)
                            │
        ┌───────────────────┼─────────────────────────┐
        ▼                   ▼                         ▼
  KBBuilder            KGBuilder                (kept on the builders)
    embeddings         merge_chunks_by_token_size
    content_blocks     extract_entities_and_relationships  (Step 1)
        │              embed_entities_and_store_to_chroma  (Step 2)
        │              store_entities_and_relationships_to_kuzu (Step 3)
        ▼                   │
  ChromaKnowledgeBase ◀─────┘
    .rebuild(chunks, embeddings)
                                                 ▼
                                          KuzuGraphStore
```

### Workflow B: Stateless KG rebuild (`UPDATE_KG=true`, `UPDATE_KB=false`)
```
ChromaKnowledgeBase.get_all()  →  {chunk_ids, documents, metadatas}
                │
                ▼
    Chunk objects → KGBuilder.merge_chunks_by_token_size
                  → extract_entities_and_relationships (LLM)
                  → embed_entities_and_store_to_chroma
                  → store_entities_and_relationships_to_kuzu
```
**Why this matters**: iterate KG-extraction logic without re-parsing PDFs or recomputing chunk embeddings.

### Workflow C: BM25 rebuild (`BUILD_BM25=true`)
```
ChromaKnowledgeBase.get_all()
        │
        ▼
ChromaKnowledgeBase.build_bm25_index_from_chroma()
        │
        ▼
bm25_index.pkl  (next to chroma.sqlite3)
```

### Workflow D: Query (`EXECUTE_QUERY=true`)
```
Query
  │
  ▼
RetrievalPipeline.run_query
  │
  ├─ Chroma.search       (vector)
  ├─ Chroma.search_bm25  (BM25)
  └─ Kuzu.search         (graph, n-hop + PageRank)
  │
  ▼
HybridFuser  (weighted_avg / rrf / max / min, normalized, deduped)
  │
  ▼
Reranker (Simple / CrossEncoder / LLM / Hybrid)
  │
  ▼
LocalAnswerGenerator → Answer
```

## Configuration

`RAGEngineConfig` aggregates the dataclasses below; every field has an env-var default. See [README.md](README.md) for the full table.

- `EmbeddingConfig`, `LLMConfig`, `VisionConfig` — model providers (OpenAI, Azure OpenAI, Gemini for vision)
- `LanguageConfig` — 7 supported languages (`en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`)
- `PDFProcessingConfig` — layout, image/table extraction, header/footer filtering
- `ProcessingConfig` — `chunker_type` (`title` or `token`), chunk sizes, worker counts, token budgets
- `BM25Config` — k1, b, language, min token length
- `HybridRetrievalConfig` — fusion strategy, weights, normalization, RRF k, dedup
- `RerankerConfig` — model selection and top-k pre/post rerank
- `StorageConfig` — cosine threshold, related-chunk count, max graph nodes

### Azure OpenAI
A single `USE_AZURE_OPENAI=true` switches LLM, embedding, and vision to Azure simultaneously. Each component has an independent override (`LLM_USE_AZURE`, `EMBEDDING_USE_AZURE`, `VISION_USE_AZURE`) and its own deployment env var.

## Key Design Decisions

### 1. Decoupled storage
**Decision**: `ChromaKnowledgeBase` and `KuzuGraphStore` are independent of pipeline and engine.
**Rationale**: enables standalone use, easy backend substitution (Pinecone, Neo4j…), and unit testing without spinning up the engine.
```python
chroma = ChromaKnowledgeBase("./chroma_db")
results = chroma.search(vector, top_k=5)   # works on its own
```

### 2. Multiple KG rebuild paths
**Decision**: KG can be rebuilt from a `Document`, from existing Chroma chunks, or from extractor output.
**Rationale**: separates expensive PDF parsing/embedding from cheap KG iteration.
```python
# From a parsed document
kg_builder.build_from_document(doc)

# From data already sitting in Chroma — no PDF reprocessing
kg_builder.rebuild_kuzu_from_chroma_chunks(...)
```

### 3. BM25 alongside vectors
**Decision**: BM25 lives next to the Chroma index and is built from the same chunks.
**Rationale**: deterministic full-text search complements the noisy semantic channel; together they yield more reliable hybrid retrieval. The index is persistent (`bm25_index.pkl`), so building it is a one-time cost.

### 4. Workflow flags in the featured example
**Decision**: `hybrid_pdf_rag_chroma_kuzu.py` exposes four boolean flags (`UPDATE_KB`, `UPDATE_KG`, `BUILD_BM25`, `EXECUTE_QUERY`).
**Rationale**: each step is independently expensive (parsing, LLM extraction, embedding, querying). Flags let you re-run just the parts that changed.

### 5. Three-step KG flow
**Decision**: explicit Step 1 (LLM extract) → Step 2 (entity embeddings → Chroma) → Step 3 (entities + relations → Kuzu).
**Rationale**: each step is observable, testable, and replaceable. Entity vectors live next to chunk vectors in Chroma so graph search can fall back to similarity when needed.

## Backwards Compatibility

The high-level `RAGEngine` API is preserved and now delegates internally to the pipeline modules:
```python
engine = RAGEngine(config)
engine.process_document("file.pdf")
engine.build_knowledge_graph_for_document(document)
engine.index_content_blocks(blocks)
engine.query("question")
```

For new code, prefer the modular pipeline API directly — it is more flexible and observable.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Document parsing | O(n) | n = text size |
| Embedding | O(n · d) | d = embedding dim, batched |
| Vector search | O(log n) | Chroma HNSW index |
| BM25 search | O(q · postings) | rank_bm25, in-memory |
| Graph search | O(e + h) | e = entities matched, h = hop expansion |
| Reranking | O(k log k) | k = candidates after fusion |

## Extension Points

- **Storage backend**: implement the `ChromaKnowledgeBase` / `KuzuGraphStore` surface and update the matching builder.
- **Retrieval / fusion**: add a method on `RetrievalPipeline` or extend `HybridFuser`.
- **KG extraction**: subclass `KnowledgeGraphBuilder` and override `extract_entities_and_relationships`.
- **Parser / processor**: add a `BaseParser` / `BaseModalProcessor` subclass and register it with the corresponding factory.

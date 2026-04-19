"""
完整 PDF Hybrid RAG 示例 - 模块化架构版本

特性：
1. 使用 `rag_engine.pipeline` 模块解析 PDF 并构建知识库和知识图谱
2. 使用 `rag_engine.storage.ChromaKnowledgeBase` 进行向量存储 (完全独立)
3. 使用 `rag_engine.storage.KuzuGraphStore` 进行图存储 (完全独立)
4. 支持三个独立的工作流：
   - UPDATE_KB: 文档处理 + 知识库构建（生成向量和 Chroma 索引）
   - UPDATE_KG: 从 Chroma 查询所有数据 + 构建知识图谱（独立执行，无需 engine）
   - EXECUTE_QUERY: 初始化 retriever + 执行查询（目前还有一些向后兼容的依赖，后续可进一步解耦）

运行方式（PowerShell）：
$env:DEEPBRICKS_API_KEY = "<your-key>"
d:/workspace/llamaindex_demo/.venv/Scripts/python.exe examples/hybrid_pdf_rag_chroma_kuzu.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from rag_engine.types import ContentBlock, ModalityType, chunks_to_content_blocks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.config import (
    EmbeddingConfig,
    LLMConfig,
    PDFProcessingConfig,
    RAGEngineConfig,
    VisionConfig,
    LanguageConfig,
)
from rag_engine.pipeline import (
    DocumentProcessor,
    KnowledgeBaseBuilder,
    KnowledgeGraphBuilder,
)
from rag_engine.storage import ChromaKnowledgeBase, KuzuGraphStore
from rag_engine.pipeline.retrieval_pipeline import (
    RetrievalPipeline,
    LocalReranker,
    LocalAnswerGenerator,
)

DEFAULT_PDF_PATH = PROJECT_ROOT / "examples" / "京东订单多维度调度系统PRD1.0.pdf"
OUTPUT_DIR = PROJECT_ROOT / "output" / "hybrid_pdf_rag_demo"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
KUZU_DIR = OUTPUT_DIR / "kuzu_db"
RESULT_PATH = OUTPUT_DIR / "last_result.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_OPENAI_BASE_URL = os.getenv("OLLAMA_OPENAI_BASE_URL", f"{OLLAMA_BASE_URL}/v1")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qllama/bge-m3:q4_k_m")
RERANK_MODEL = os.getenv("OLLAMA_RERANK_MODEL", "qllama/bge-reranker-v2-m3:q4_k_m")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "GPT-4o")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "openai")
DEEPBRICKS_BASE_URL = os.getenv("DEEPBRICKS_BASE_URL", "https://api.deepbricks.ai/v1/")
DEEPBRICKS_API_KEY = os.getenv("DEEPBRICKS_API_KEY", "sk-F4x4h7mc8GjjURTPcxGMXnuuo463D1p2clJ4Pp55ch00QH30") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPDATE_KB = os.getenv("UPDATE_KB", "false").strip().lower() == "true"
UPDATE_KG = os.getenv("UPDATE_KG", "false").strip().lower() == "true"
EXECUTE_QUERY = os.getenv("EXECUTE_QUERY", "true").strip().lower() == "true"
MARKDOWN_PATH = os.getenv("MARKDOWN_PATH", "aaa.md")
# MARKDOWN_PATH = os.getenv("MARKDOWN_PATH")

def configure_console() -> None:
    """Force UTF-8 output on Windows terminals when available."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_config() -> RAGEngineConfig:
    """Build RAG engine configuration."""
    vision_config = VisionConfig(
        model=VISION_MODEL,
        api_key=DEEPBRICKS_API_KEY if DEEPBRICKS_BASE_URL else (GEMINI_API_KEY or DEEPBRICKS_API_KEY),
        base_url=DEEPBRICKS_BASE_URL,
        enabled=True,
        provider=VISION_PROVIDER,
    )
    
    return RAGEngineConfig(
        working_dir=str(OUTPUT_DIR),
        output_dir=str(OUTPUT_DIR),
        embedding=EmbeddingConfig(
            model=EMBED_MODEL,
            dimension=1024,
            api_key="ollama",
            base_url=OLLAMA_OPENAI_BASE_URL,
            batch_num=32,
            max_async=10,
        ),
        llm=LLMConfig(
            model=GEN_MODEL,
            api_key=DEEPBRICKS_API_KEY,
            base_url=DEEPBRICKS_BASE_URL,
            temperature=0.2,
            max_tokens=900,
            model_max_token_size=8192,
            model_max_async=10,
            enable_cache=True,
        ),
        vision=vision_config,
        language=LanguageConfig(default_language="zh"),
        pdf_processing=PDFProcessingConfig.from_vision_config(
            vision_config,
            use_advanced_layout=True,
            extract_images=True,
            extract_tables=True,
            extract_text=True,
            use_vision_api=True,
        ),
        log_level="INFO",
    )


# ============================================================================
# WORKFLOW 1: UPDATE_KB - Document processing + KB construction
# ============================================================================

def build_knowledge_base_step1(pdf_path: Path, config: RAGEngineConfig) -> Dict[str, Any]:
    """
    步骤 1: 处理 PDF 文档，构建知识库
    
    Responsibilities:
    - Parse PDF and extract content blocks
    - Build vector embeddings
    - Index into Chroma KB
    """
    print("\n" + "=" * 88)
    print("[UPDATE_KB] 步骤 1: 文档处理和知识库构建")
    print("=" * 88)
    
    # Initialize processors
    doc_processor = DocumentProcessor(config)
    kb_builder = KnowledgeBaseBuilder(config)
    
    # Parse document
    print(f"解析 PDF: {pdf_path}")
    document = doc_processor.process_document(str(pdf_path), language="zh", markdown_path=MARKDOWN_PATH)
    print(f"✓ 已生成 {len(document.chunks)} 个 chunks")
    
    # Build knowledge base (embeddings + Chroma index)
    print("\n生成向量嵌入和索引...")
    kb_builder.build_from_document(document)
    print(f"✓ 已生成 {len(kb_builder.get_embeddings())} 个向量嵌入")
    
    # Rebuild Chroma
    print("\n重建 Chroma 向量数据库...")
    chroma_kb = ChromaKnowledgeBase(CHROMA_DIR)
    kb_builder.rebuild_chroma(chroma_kb)
    print(f"✓ Chroma 向量块数: {chroma_kb.count()}")
    
    return {
        "document": document,
        "kb_builder": kb_builder,
        "chroma_kb": chroma_kb,
        "chunks": document.chunks,
        "embeddings": kb_builder.get_embeddings(),
    }

def build_knowledge_graph_step1(kb_data: Dict[str, Any], config: RAGEngineConfig) -> None:
    """
    步骤 2: 构建知识图谱 - 三步骤流程
    
    Workflow:
    Step 1: 通过LLM提取实体和实体关系
    Step 2: 实体向量化并插入Chroma
    Step 3: 实体和关系插入Kuzu图数据库
    """
    print("\n" + "=" * 88)
    print("[UPDATE_KB] 步骤 2: 知识图谱构建 (三步骤流程)")
    print("=" * 88)
    
    kg_builder = KnowledgeGraphBuilder(config)
    chroma_kb = kb_data["chroma_kb"]
    
    # ========== Step 1: 通过LLM提取实体和关系 ==========
    print("\n[Step 1] 通过LLM提取实体和关系...")
    content_blocks=kg_builder.merge_chunks_by_token_size(kb_data["chunks"])
    entities, relationships = kg_builder.extract_entities_and_relationships(
        content_blocks
    )
    print(f"✓ 已提取 {len(entities)} 个实体和 {len(relationships)} 个关系")
    
    # ========== Step 2: 实体向量化并插入Chroma ==========
    print("\n[Step 2] 实体向量化并存储到Chroma...")
    kg_builder.embed_entities_and_store_to_chroma(entities, chroma_kb)
    print("✓ 实体向量已存储到Chroma")
    
    # ========== Step 3: 实体和关系插入Kuzu ==========
    print("\n[Step 3] 实体和关系存储到Kuzu图数据库...")
    kuzu_store = KuzuGraphStore(KUZU_DIR)
    kg_builder.store_entities_and_relationships_to_kuzu(kuzu_store, kb_data["content_blocks"])
    print("✓ 实体和关系已存储到Kuzu")
    
    # Print statistics
    stats = kg_builder.get_graph_stats()
    print(f"\n知识图谱统计: {stats['entities']} 个实体 | {stats['relationships']} 个关系")


# ============================================================================
# WORKFLOW 2: UPDATE_KG - Independent KG rebuilding from Chroma
# ============================================================================

def rebuild_knowledge_graph_from_chroma() -> None:
    """
    步骤 2b (独立): 从 Chroma 查询所有数据，重建知识图谱
    
    This is a STATELESS operation using the three-step workflow:
    Step 1: 从 Chroma chunks 提取实体和关系
    Step 2: 实体向量化并存储到 Chroma
    Step 3: 实体和关系存储到 Kuzu
    """
    print("\n" + "=" * 88)
    print("[UPDATE_KG] 从 Chroma 独立重建知识图谱 (三步骤流程)")
    print("=" * 88)
    
    config = build_config()
    chroma_kb = ChromaKnowledgeBase(CHROMA_DIR)
    
    # Query all chunks from Chroma
    print("\n[Step 1] 从 Chroma 查询所有 chunks 并提取实体/关系...")
    try:
        all_data = chroma_kb.get_all()
        chunk_ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        if not chunk_ids:
            print("⚠️  Chroma 中没有数据，无法重建知识图谱")
            return
        
        print(f"✓ 从 Chroma 查询到 {len(chunk_ids)} 个 chunks")
        
        # Convert Chroma chunks to Chunk objects first
        print("转换 chunks 为 Chunk 对象...")
        from rag_engine.types import Chunk, ContentBlock, ContentType, ModalityType
        
        chunk_objects = []
        for chunk_id, text, meta in zip(chunk_ids, documents, metadatas):
            chunk = Chunk(
                text=text,
                chunk_type=meta.get("content_type", "text"),
                id=chunk_id,
                metadata=meta
            )
            chunk_objects.append(chunk)
        
        print(f"✓ 转换了 {len(chunk_objects)} 个 Chunk 对象")
        
        # Merge chunks by token size
        print("\n合并 chunks 按 token 大小...")
        kg_builder = KnowledgeGraphBuilder(config)
        merged_chunks = kg_builder.merge_chunks_by_token_size(chunk_objects)
        print(f"✓ 合并后得到 {len(merged_chunks)} 个 chunks")
        
        # Convert merged Chunk objects to ContentBlock objects for entity extraction
        print("转换合并后的 chunks 为 ContentBlock 对象...")
        content_blocks = []
        for chunk in merged_chunks:
            content_type = ContentType(chunk.chunk_type)
            
            # Determine modality based on content type
            if content_type in [ContentType.IMAGE, ContentType.CHART, ContentType.DIAGRAM]:
                modality = ModalityType.VISUAL
            elif content_type in [ContentType.TABLE, ContentType.EQUATION, ContentType.CODE]:
                modality = ModalityType.STRUCTURED
            else:
                modality = ModalityType.TEXT
            
            block = ContentBlock(
                id=chunk.id,
                type=content_type,
                content=chunk.text,
                modality=modality,
                page_num=chunk.metadata.get("page_num", 0),
                language=chunk.metadata.get("language", "zh"),
                source_file=chunk.metadata.get("source_file", ""),
                metadata=chunk.metadata
            )
            content_blocks.append(block)
        
        print(f"✓ 转换了 {len(content_blocks)} 个 ContentBlock 对象")
        
        # ========== Step 1: 通过LLM提取实体和关系 ==========
        print("\n通过 LLM 提取实体和关系...")
        entities, relationships = kg_builder.extract_entities_and_relationships(content_blocks)
        print(f"✓ 已提取 {len(entities)} 个实体和 {len(relationships)} 个关系")
        
        # ========== Step 2: 实体向量化并插入Chroma ==========
        print("\n[Step 2] 实体向量化并存储到 Chroma...")
        kg_builder.embed_entities_and_store_to_chroma(entities, chroma_kb)
        print("✓ 实体向量已存储到 Chroma")
        
        # ========== Step 3: 实体和关系插入Kuzu ==========
        print("\n[Step 3] 实体和关系存储到 Kuzu...")
        kuzu_store = KuzuGraphStore(KUZU_DIR)
        kg_builder.store_entities_and_relationships_to_kuzu(
            kuzu_store, 
            {block.id: block for block in content_blocks}
        )
        print("✓ 实体和关系已存储到 Kuzu")
        
        # Print statistics
        stats = kg_builder.get_graph_stats()
        print(f"\n✓ 知识图谱重建完成: {stats['entities']} 个实体 | {stats['relationships']} 个关系")
        
    except Exception as e:
        print(f"❌ 从 Chroma 重建知识图谱失败: {e}")
        raise


# ============================================================================
# WORKFLOW 3: EXECUTE_QUERY - Query execution
# ============================================================================

async def execute_query_workflow(pdf_path: Path) -> None:
    """
    执行查询工作流：
    1. 使用 RetrievalPipeline 内部矢量化查询
    2. 执行混合检索 (vector + graph)
    3. Rerank 结果
    4. 生成答案
    """
    print("\n" + "=" * 88)
    print("[EXECUTE_QUERY] 执行查询工作流")
    print("=" * 88)
    
    config = build_config()
    
    # Load data from storage
    chroma_kb = ChromaKnowledgeBase(CHROMA_DIR)
    with KuzuGraphStore(KUZU_DIR) as kuzu_store:
        # Check if Chroma has data
        if chroma_kb.count() == 0:
            print("⚠️  Chroma KB 中无数据，跳过查询")
            return
        
        retrieval_pipeline = RetrievalPipeline(config)
        reranker = LocalReranker(OLLAMA_BASE_URL, RERANK_MODEL)
        generator = LocalAnswerGenerator(
            base_url=DEEPBRICKS_BASE_URL,
            api_key=DEEPBRICKS_API_KEY,
            model=GEN_MODEL,
        )
        
        # Run queries
        queries = [
            "订单已经被调度过，是否可以修改订单？",
        ]
        
        outputs = []
        for idx, query in enumerate(queries, 1):
            print(f"\n--- Query {idx}: {query}")
            
            # Run full retrieval pipeline
            result = await retrieval_pipeline.run_query(
                query=query,
                chroma_kb=chroma_kb,
                kuzu_store=kuzu_store,
                reranker=reranker,
                generator=generator,
                top_k=5,
            )
            
            outputs.append(result)
            
            print("回答：")
            print(result["answer"])
            print("Top-3 证据片段：")
            for doc in result["top_docs"][:3]:
                channel = doc.get("channel", "hybrid")
                score = doc.get("score", 0.0)
                preview = doc.get("preview", "")
                print(f"  - [{channel}] score={score:.4f} | {preview}")
        
        # Save results
        payload = {
            "pdf_path": str(pdf_path),
            "models": {
                "embedding": EMBED_MODEL,
                "reranker": RERANK_MODEL,
                "generator": GEN_MODEL,
            },
            "results": outputs,
        }
        
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 结果已保存到: {RESULT_PATH}")


# ============================================================================
# Main orchestration
# ============================================================================

def main() -> None:
    configure_console()

    pdf_path = Path(os.getenv("RAG_PDF_PATH", str(DEFAULT_PDF_PATH))).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 88)
    print("Hybrid PDF RAG Demo - 模块化架构")
    print("=" * 88)
    print(f"PDF 数据源: {pdf_path}")
    print(f"Embedding 模型: {EMBED_MODEL}")
    print(f"Rerank 模型: {RERANK_MODEL}")
    print(f"生成模型: {GEN_MODEL}")
    print("\n执行标志:")
    print(f"  UPDATE_KB (文档处理+知识库+知识图谱): {UPDATE_KB}")
    print(f"  UPDATE_KG (从 Chroma 重建知识图谱):   {UPDATE_KG}")
    print(f"  EXECUTE_QUERY (执行查询):           {EXECUTE_QUERY}")
    
    # Check existing data
    chroma_exists = (CHROMA_DIR / "chroma.sqlite3").exists() or (CHROMA_DIR / "index").exists()
    kuzu_exists = (KUZU_DIR / "kuzu.db").exists()
    
    print(f"\n数据库状态:")
    print(f"  Chroma DB 存在: {chroma_exists}")
    print(f"  Kuzu DB 存在: {kuzu_exists}")

    # ========== WORKFLOW EXECUTION ==========
    config = build_config()
    
    # WORKFLOW 1: UPDATE_KB
    if UPDATE_KB:
        kb_data = build_knowledge_base_step1(pdf_path, config)
        build_knowledge_graph_step1(kb_data, config)
    elif UPDATE_KG or EXECUTE_QUERY:
        if not chroma_exists:
            print("\n⚠️  Chroma DB 不存在，无法继续")
            print("   请先执行 UPDATE_KB=true")
            return
    
    # WORKFLOW 2: UPDATE_KG (independent)
    if not UPDATE_KB and UPDATE_KG:
        rebuild_knowledge_graph_from_chroma()
    elif EXECUTE_QUERY:
        if not kuzu_exists:
            print("\n⚠️  Kuzu DB 不存在")
            print("   将跳过图数据库检索")
    
    # WORKFLOW 3: EXECUTE_QUERY
    if EXECUTE_QUERY:
        asyncio.run(execute_query_workflow(pdf_path))
    else:
        print("\n⏭️  未执行查询 (EXECUTE_QUERY=false)")


if __name__ == "__main__":
    main()

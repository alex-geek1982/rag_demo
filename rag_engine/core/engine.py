"""
RAG Engine - Main RAG engine orchestrator (simplified coordinating layer)

Architecture Overview:
- engine.py (this file): Coordination layer - delegates to specialized modules
- pipeline/: Document processing, KB building, KG building, retrieval
- storage/: Chroma vector DB and Kuzu graph DB operations (completely decoupled)
- retrieval/: Embedding providers and retrieval utilities

Key Design Principles:
1. Single Responsibility: Each component has one clear purpose
2. Modularity: Can be used independently or combined
3. Decoupling: Storage layers don't depend on engine state
4. Flexibility: Rebuild KB/KG independently without reinitializing engine
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json
from datetime import datetime
import asyncio

from ..config import RAGEngineConfig
from ..types import Document, ContentBlock, QueryResult, RetrievalResult, Entity
from ..i18n import get_i18n

# New modular components
from ..pipeline import (
    DocumentProcessor,
    KnowledgeBaseBuilder,
    KnowledgeGraphBuilder,
    RetrievalPipeline,
)
from ..pipeline.retrieval_pipeline import LocalAnswerGenerator
from ..storage import ChromaKnowledgeBase, KuzuGraphStore
from ..retrieval import OpenAIEmbedding, HybridRetriever
from ..retrieval.reranker import SimpleReranker, LLMReranker, CrossEncoderReranker, HybridReranker
from ..core.knowledge_graph import KnowledgeGraph, EntityExtractor, RelationshipBuilder

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Simplified RAG Engine - Orchestrator for modular components.
    
    Main responsibilities:
    - Coordinate document processing, KB building, KG building
    - Provide backward-compatible API
    - Manage configuration and component lifecycle
    
    Implementation note: Delegates actual work to specialized modules in pipeline/ and storage/
    """
    
    def __init__(self, config: Optional[RAGEngineConfig] = None):
        """
        Initialize RAG engine with modular architecture.
        
        Args:
            config: RAG engine configuration
        """
        self.config = config or RAGEngineConfig.from_env()
        self.i18n = get_i18n(embedding_provider=None)
        
        # Core components (backward compatibility)
        self.embedding_provider = None
        self.retriever = None
        self.kg = None
        self.entity_extractor = None
        self.relationship_builder = None
        
        # New modular components
        self.document_processor = DocumentProcessor(self.config)
        self.kb_builder = KnowledgeBaseBuilder(self.config)
        self.kg_builder = KnowledgeGraphBuilder(self.config)
        self.retrieval_pipeline = RetrievalPipeline(self.config, self.content_blocks_dict)
        
        # Storage (not coupled with this engine)
        self.chroma_kb = None
        self.kuzu_store = None
        self.entities: Dict[str, Entity] = {}
        self.content_blocks_dict: Dict[str, ContentBlock] = {}
        
        # Storage
        self.documents: Dict[str, Document] = {}
        self.processed_blocks: List[ContentBlock] = []
        
        logger.info("RAG Engine initialized (modular architecture)")
    
    def _init_embeddings(self) -> None:
        """Initialize embedding provider (backward compatibility)."""
        if self.embedding_provider is not None:
            return
            
        if not self.config.embedding.api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.embedding_provider = OpenAIEmbedding(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            base_url=self.config.embedding.base_url
        )
    
    def _init_retriever(self) -> None:
        """Initialize retriever (backward compatibility)."""
        if self.retriever is not None:
            return
            
        self._init_embeddings()
        self.retriever = HybridRetriever(
            embedding_provider=self.embedding_provider,
            embedding_dim=self.config.embedding.dimension,
            language=self.config.language.default_language
        )
    
    def _init_knowledge_graph(self) -> None:
        """Initialize knowledge graph (backward compatibility)."""
        if self.kg is not None:
            return
            
        self.kg = KnowledgeGraph("main")
        self.entity_extractor = EntityExtractor(self.config.language.default_language)
        self.relationship_builder = RelationshipBuilder(self.config.language.default_language)
    
    def process_document(self, file_path: str, doc_id: Optional[str] = None, doc_title: Optional[str] = None, language: Optional[str] = None) -> Document:
        """
        Process a single document (delegate to DocumentProcessor).
        
        Args:
            file_path: Path to document file
            doc_id: Optional document ID
            doc_title: Optional document title
            language: Optional document language
        
        Returns:
            Processed document with content blocks
        """
        document = self.document_processor.process_document(
            file_path, doc_id, doc_title, language
        )
        
        # Store for backward compatibility
        self.documents[document.id] = document
        self.processed_blocks.extend(document.content_blocks)
        self.content_blocks_dict.update({block.id: block for block in document.content_blocks})
        
        return document
    
    async def process_document_async(self, file_path: str, doc_id: Optional[str] = None, doc_title: Optional[str] = None, language: Optional[str] = None) -> Document:
        """Asynchronous version of process_document"""
        return await asyncio.to_thread(
            self.process_document,
            file_path,
            doc_id,
            doc_title,
            language
        )
    
    def build_knowledge_graph_for_document(self, document: Document) -> None:
        """
        Build knowledge graph for a document (delegate to KnowledgeGraphBuilder).
        
        Args:
            document: Document to build graph for
        """
        self.kg_builder.build_from_document(document)
        
        # Sync to legacy kg for backward compatibility
        self._init_knowledge_graph()
        self.kg = self.kg_builder.kg
        if self.kg:
            self.entity_extractor = EntityExtractor(self.config.language.default_language)
            self.relationship_builder = RelationshipBuilder(self.config.language.default_language)
    
    async def build_knowledge_graph_lightsag_style(self, content_blocks: List[ContentBlock], 
                                                 llm_func=None, global_config=None) -> None:
        """
        Build knowledge graph using LightRAG approach with LLM-based extraction.
        
        Args:
            content_blocks: List of content blocks to build graph from
            llm_func: LLM function for entity/relationship extraction
            global_config: Global configuration for prompts and settings
        """
        logger.info(f"Building knowledge graph using LightRAG approach from {len(content_blocks)} content blocks")
        
        # Use LightRAG-style builder
        await self.kg_builder.build_from_blocks_lightsag_style(
            content_blocks, llm_func, global_config
        )
        
        # Rebuild Kuzu graph from extracted data
        if self.kuzu_store:
            self.kg_builder.rebuild_kuzu_from_extracted_data(self.kuzu_store)
        
        logger.info("Knowledge graph built using LightRAG approach")
    
    def index_content_blocks(self, content_blocks: List[ContentBlock]) -> None:
        """
        Index content blocks to retriever (backward compatibility).
        
        Args:
            content_blocks: Content blocks to index
        """
        try:
            self._init_retriever()
            self.retriever.add_content_sync(content_blocks)
            self.content_blocks_dict.update({block.id: block for block in content_blocks})
            logger.info(f"Indexed {len(content_blocks)} content blocks to retriever")
        except Exception as e:
            logger.error(f"Failed to index content blocks: {e}")
            raise
    
    def process_folder(self, folder_path: str, language: Optional[str] = None, recursive: bool = True) -> List[Document]:
        """
        Process multiple documents in a folder.
        
        Args:
            folder_path: Path to folder
            language: Document language
            recursive: Whether to process subfolders
        
        Returns:
            List of processed documents
        """
        documents = self.document_processor.process_folder(folder_path, language, recursive)
        
        for doc in documents:
            self.documents[doc.id] = doc
            self.processed_blocks.extend(doc.content_blocks)
            self.content_blocks_dict.update({block.id: block for block in doc.content_blocks})
        
        return documents
    
    def _process_content_blocks(self, document: Document) -> None:
        """Process content blocks (for backward compatibility)."""
        # Actually delegated to document_processor
        pass

    def attach_storage(
        self,
        chroma_db_path: str,
        kuzu_db_path: str,
        content_blocks: Optional[Dict[str, ContentBlock]] = None,
        entities: Optional[Dict[str, Entity]] = None,
    ) -> None:
        """Attach Chroma and Kuzu storage to the engine for graph RAG queries."""
        self.chroma_kb = ChromaKnowledgeBase(Path(chroma_db_path))
        self.kuzu_store = KuzuGraphStore(Path(kuzu_db_path))
        self.content_blocks_dict = (
            content_blocks
            if content_blocks is not None
            else {block.id: block for block in self.processed_blocks}
        )
        if entities is not None:
            self.entities = entities

    def _build_content_blocks_map(self) -> Dict[str, ContentBlock]:
        if not self.content_blocks_dict and self.processed_blocks:
            self.content_blocks_dict = {block.id: block for block in self.processed_blocks}
        return self.content_blocks_dict

    def _create_llm_rerank_func(self):
        def llm_model_func(prompt: str, _priority: Optional[int] = None) -> str:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.config.llm.api_key, base_url=self.config.llm.base_url)
                response = client.chat.completions.create(
                    model=self.config.llm.model,
                    temperature=0.0,
                    max_tokens=32,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a relevance ranking assistant. Score each document from 0.0 to 1.0 in a comma-separated list.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM rerank function failed: {e}")
                return ""

        return llm_model_func

    def _create_reranker(self):
        model_type = self.config.reranker.rerank_model.lower()

        if model_type == "cross-encoder":
            return CrossEncoderReranker(
                self.config.reranker.cross_encoder_model,
                self.config.reranker.device,
            )
        if model_type == "llm":
            return LLMReranker(
                llm_model_func=self._create_llm_rerank_func(),
                batch_size=self.config.reranker.batch_size,
            )
        if model_type == "hybrid":
            return HybridReranker(
                primary_reranker=SimpleReranker(),
                secondary_reranker=LLMReranker(
                    llm_model_func=self._create_llm_rerank_func(),
                    batch_size=self.config.reranker.batch_size,
                ),
                primary_weight=0.7,
                secondary_weight=0.3,
            )

        return SimpleReranker()

    def _run_graph_rag_query(
        self,
        query: str,
        top_k: int,
        query_language: str,
    ) -> QueryResult:
        if not self.chroma_kb or not self.kuzu_store:
            raise RuntimeError(
                "ChromaKnowledgeBase and KuzuGraphStore must be attached before running a graph RAG query."
            )

        content_blocks = self._build_content_blocks_map()
        entities = self.entities or {}
        combined = self.retrieval_pipeline.retrieve_hybrid(
            query,
            self.chroma_kb,
            self.kuzu_store,
            entities,
            content_blocks,
            top_k,
        )

        # Separate graph results from text chunks - only rerank text chunks
        text_chunks = [doc for doc in combined if not doc.metadata.get("retrieval_channel", "").startswith("graph/kuzu")]
        graph_results = [doc for doc in combined if doc.metadata.get("retrieval_channel", "").startswith("graph/kuzu")]

        reranker = self._create_reranker() if self.config.reranker.enable_rerank else None
        if reranker:
            reranked_chunks = reranker.rerank(query, text_chunks, top_k)
            # Combine reranked chunks with graph results
            reranked = self.retrieval_pipeline._merge_results(reranked_chunks, graph_results)[:top_k]
        else:
            reranked = (text_chunks + graph_results)[:top_k]

        generator = LocalAnswerGenerator(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        try:
            answer = generator.generate(query, reranked)
        except Exception as e:
            logger.warning(f"Graph RAG answer generation failed: {e}")
            answer = self._generate_answer(query, reranked)

        confidence = sum(doc.score for doc in reranked) / len(reranked) if reranked else 0.0

        return QueryResult(
            query=query,
            answer=answer,
            retrieved_docs=reranked,
            sources=[doc.doc_id for doc in reranked],
            confidence=float(confidence),
            metadata={
                "query_language": query_language,
                "retrieved_languages": list({
                    doc.metadata.get("language", "unknown")
                    for doc in reranked
                }),
                "cross_lingual_retrieval": any(
                    doc.metadata.get("cross_lingual", False) for doc in reranked
                ),
                "retrieval_channel": "hybrid_graph",
            },
        )
    
    def _build_knowledge_graph(self, document: Document) -> None:
        """Build knowledge graph (for backward compatibility)."""
        self.build_knowledge_graph_for_document(document)
    
    def rebuild_knowledge_graph_from_blocks(self, content_blocks: List[ContentBlock]) -> None:
        """
        Rebuild knowledge graph from content blocks (stateless method).
        
        Args:
            content_blocks: List of content blocks to build graph from
        """
        logger.info(f"Rebuilding knowledge graph from {len(content_blocks)} content blocks")
        self.kg_builder.build_from_blocks(content_blocks)
        
        # Sync to legacy kg for backward compatibility
        self._init_knowledge_graph()
        self.kg = self.kg_builder.kg
    
    def query(self, query: str, top_k: int = 5, use_graph: bool = True, query_language: Optional[str] = None) -> QueryResult:
        """
        Synchronous query with multilingual support.
        
        If graph storage is attached, performs hybrid graph-based retrieval and LLM answer generation.
        Otherwise falls back to embedding-based retrieval.
        """
        try:
            self._init_retriever()
            
            from ..i18n import LanguageDetector
            
            logger.info("Processing query with multilingual support")
            
            if query_language is None:
                detector = LanguageDetector()
                query_language = detector.detect(query)
            
            logger.debug(f"Query language detected: {query_language}")

            if use_graph and self.chroma_kb and self.kuzu_store:
                return self._run_graph_rag_query(query, top_k, query_language)

            retrieved_docs = self.retriever.retrieve_sync(
                query, 
                top_k=top_k,
                query_language=query_language
            )
            
            answer = self._generate_answer(query, retrieved_docs)
            
            confidence = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            
            return QueryResult(
                query=query,
                answer=answer,
                retrieved_docs=retrieved_docs,
                sources=[doc.doc_id for doc in retrieved_docs],
                confidence=float(confidence),
                metadata={
                    "query_language": query_language,
                    "retrieved_languages": list(set(
                        doc.metadata.get("language", "unknown") 
                        for doc in retrieved_docs
                    )) if retrieved_docs else [],
                    "cross_lingual_retrieval": any(
                        doc.metadata.get("cross_lingual", False) 
                        for doc in retrieved_docs
                    ) if retrieved_docs else False
                }
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    async def aquery(self, query: str, top_k: int = 5, use_graph: bool = True, query_language: Optional[str] = None) -> QueryResult:
        """
        Asynchronous query with multilingual support.
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_graph: Whether to use knowledge graph
            query_language: Optional language of query
        
        Returns:
            Query result with multilingual support
        """
        try:
            self._init_retriever()
            
            from ..i18n import LanguageDetector
            
            logger.info("Processing query with multilingual support")
            
            if query_language is None:
                detector = LanguageDetector()
                query_language = detector.detect(query)
            
            logger.debug(f"Query language detected: {query_language}")

            if use_graph and self.chroma_kb and self.kuzu_store:
                return self._run_graph_rag_query(query, top_k, query_language)

            retrieved_docs = await self.retriever.retrieve(
                query, 
                top_k=top_k,
                query_language=query_language
            )
            
            answer = self._generate_answer(query, retrieved_docs)
            
            confidence = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            
            return QueryResult(
                query=query,
                answer=answer,
                retrieved_docs=retrieved_docs,
                sources=[doc.doc_id for doc in retrieved_docs],
                confidence=float(confidence),
                metadata={
                    "query_language": query_language,
                    "retrieved_languages": list(set(
                        doc.metadata.get("language", "unknown") 
                        for doc in retrieved_docs
                    )) if retrieved_docs else [],
                    "cross_lingual_retrieval": any(
                        doc.metadata.get("cross_lingual", False) 
                        for doc in retrieved_docs
                    ) if retrieved_docs else False
                }
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def _generate_answer(self, query: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Generate answer from retrieved documents (for backward compatibility)."""
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
        
        # For now, return concatenated context
        # In a full implementation, this would use an LLM
        context = "\n".join([f"[{doc.content_type.value}] {doc.content[:300]}" for doc in retrieved_docs])
        
        answer = f"Based on the retrieved information:\n\n{context}\n\n[Note: Full LLM integration is needed for complete answer generation]"
        return answer
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics (for backward compatibility)."""
        kg_stats = self.kg.get_stats() if self.kg else {}
        retriever_stats = self.retriever.get_retrieval_stats() if self.retriever else {}
        
        return {
            "processed_documents": len(self.documents),
            "total_content_blocks": len(self.processed_blocks),
            "knowledge_graph": kg_stats,
            "retriever": retriever_stats,
            "config": {
                "language": self.config.language.default_language,
                "embedding_model": self.config.embedding.model,
                "llm_model": self.config.llm.model,
            }
        }
    
    def save_state(self, output_dir: Optional[str] = None) -> str:
        """Save engine state (for backward compatibility)."""
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save knowledge graph
        if self.kg:
            kg_path = output_path / "knowledge_graph.json"
            self.kg.save(str(kg_path))
        
        # Save documents metadata
        docs_path = output_path / "documents.json"
        docs_data = {
            doc_id: {
                "title": doc.title,
                "source": doc.source_path,
                "blocks_count": len(doc.content_blocks),
                "created_at": doc.created_at.isoformat(),
            }
            for doc_id, doc in self.documents.items()
        }
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats_path = output_path / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_statistics(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Engine state saved to: {output_path}")
        return str(output_path)


def create_engine(config: Optional[RAGEngineConfig] = None) -> RAGEngine:
    """Factory function to create RAG engine"""
    return RAGEngine(config)

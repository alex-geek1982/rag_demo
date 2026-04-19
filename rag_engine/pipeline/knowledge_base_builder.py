"""
Knowledge Base Builder - Build and manage vector knowledge base
"""
import logging
from typing import Dict, List, Any

from ..types import Document, Chunk
from ..config import RAGEngineConfig
from ..retrieval import OpenAIEmbedding
from ..storage import ChromaKnowledgeBase

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:
    """
    Responsibility: Build and manage the vector knowledge base.
    
    This class handles:
    - Embedding generation for content chunks
    - Vector knowledge base creation and indexing
    - Independent rebuild from documents or content chunks
    
    It is independent of graph building or query execution.
    """

    def __init__(self, config: RAGEngineConfig):
        """
        Initialize knowledge base builder.
        
        Args:
            config: RAG engine configuration
        """
        self.config = config
        self.embedding_provider = None
        self.chunks: Dict[str, Chunk] = {}
        self.embeddings: Dict[str, List[float]] = {}

    def _init_embeddings(self) -> None:
        """Initialize embedding provider (lazy)."""
        if self.embedding_provider is not None:
            return

        if not self.config.embedding.api_key:
            raise ValueError("OpenAI API key not configured")

        self.embedding_provider = OpenAIEmbedding(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            base_url=self.config.embedding.base_url,
        )

    def build_from_document(self, document: Document) -> None:
        """
        Build knowledge base from a single document.
        
        Args:
            document: Document with content chunks to build from
        """
        logger.info(f"Building KB from document: {document.title}")
        self.build_from_chunks(document.chunks)

    def build_from_chunks(self, chunks: List[Chunk]) -> None:
        """
        Build knowledge base from content chunks.
        
        Args:
            chunks: List of content chunks
        """
        logger.info(f"Building KB from {len(chunks)} chunks")

        try:
            self._init_embeddings()

            # Store chunks
            self.chunks = {chunk.id: chunk for chunk in chunks}

            # Generate embeddings in batches
            chunk_ids = [chunk.id for chunk in chunks]
            texts = [chunk.text for chunk in chunks]

            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embedding_vectors = self.embedding_provider.embed_text_sync(texts)

            # Map embeddings to chunk IDs
            for chunk_id, vector in zip(chunk_ids, embedding_vectors):
                self.embeddings[chunk_id] = vector.tolist()

            logger.info(f"Generated {len(self.embeddings)} embeddings")

        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            raise

    def rebuild_chroma(self, chroma_kb: ChromaKnowledgeBase) -> None:
        """
        Rebuild Chroma knowledge base with current chunks and embeddings.
        
        Args:
            chroma_kb: ChromaKnowledgeBase instance
            collection_name: Name of collection to create
        """
        if not self.chunks or not self.embeddings:
            raise RuntimeError("No chunks or embeddings available. Call build_from_chunks first.")

        logger.info("Rebuilding Chroma knowledge base...")
        chroma_kb.rebuild(self.chunks, self.embeddings)

    def get_embeddings(self) -> Dict[str, List[float]]:
        """Get all generated embeddings."""
        return self.embeddings.copy()

    def get_chunks(self) -> Dict[str, Chunk]:
        """Get all chunks."""
        return self.chunks.copy()

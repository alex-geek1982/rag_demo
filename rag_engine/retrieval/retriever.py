"""
RAG Engine - Core retrieval system with reranking and context extraction
"""
from typing import List, Dict, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import logging
import numpy as np
from collections import defaultdict
import asyncio

from ..types import RetrievalResult, ContentType, Entity, ContentBlock
from ..i18n import get_i18n


logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers"""
    
    @abstractmethod
    async def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Embed text documents
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Embedding matrix (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_text_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous version of embed_text"""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI / Azure OpenAI embedding provider"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        base_url: Optional[str] = None,
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        """Initialize embedding provider (supports OpenAI and Azure OpenAI)."""
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        # On Azure, the "model" parameter must be the deployment name.
        self.azure_deployment = azure_deployment or model

    def _build_async_client(self):
        if self.use_azure:
            from openai import AsyncAzureOpenAI
            return AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_sync_client(self):
        if self.use_azure:
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _model_name(self) -> str:
        return self.azure_deployment if self.use_azure else self.model

    async def embed_text(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI / Azure OpenAI API"""
        try:
            client = self._build_async_client()
            response = await client.embeddings.create(
                model=self._model_name(),
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except ImportError:
            logger.error("openai library not installed. Install with: pip install openai")
            raise

    def embed_text_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous embedding using OpenAI / Azure OpenAI API"""
        try:
            client = self._build_sync_client()
            response = client.embeddings.create(
                model=self._model_name(),
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except ImportError:
            logger.error("openai library not installed. Install with: pip install openai")
            raise


class HybridRetriever:
    """Hybrid retriever combining vector and graph-based search with reranking"""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        embedding_dim: int = 3072,
        language: str = "en",
        reranker: Optional[Any] = None,
        context_extractor: Optional[Any] = None,
        enable_rerank: bool = False,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_provider: Embedding provider
            embedding_dim: Embedding dimension
            language: Query language
            reranker: Optional reranker for result re-ranking
            context_extractor: Optional context extractor for multimodal content
            enable_rerank: Whether to enable reranking
        """
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_dim
        self.language = language
        self.i18n = get_i18n()
        self.reranker = reranker
        self.context_extractor = context_extractor
        self.enable_rerank = enable_rerank
        
        # In-memory storage for now
        self.embeddings: Dict[str, np.ndarray] = {}
        self.content_blocks: Dict[str, ContentBlock] = {}
        self.entities: Dict[str, Entity] = {}
        self.modality_index: Dict[ContentType, List[str]] = defaultdict(list)
    
    async def add_content(self, blocks: List[ContentBlock]) -> None:
        """
        Add content blocks for indexing
        
        Args:
            blocks: List of content blocks
        """
        try:
            # Extract text for embedding
            texts = []
            block_ids = []
            
            for block in blocks:
                self.content_blocks[block.id] = block
                self.modality_index[block.type].append(block.id)
                
                # For non-text content, use description
                if block.type == ContentType.TEXT:
                    texts.append(block.content)
                else:
                    texts.append(f"{block.type.value}: {block.content[:500]}")
                
                block_ids.append(block.id)
            
            if texts:
                # Embed all texts
                embeddings = await self.embedding_provider.embed_text(texts)
                
                for bid, embedding in zip(block_ids, embeddings):
                    self.embeddings[bid] = embedding
                    self.content_blocks[bid].embeddings = embedding.tolist()
                
                logger.info(f"Indexed {len(blocks)} content blocks")
        except Exception as e:
            logger.error(f"Failed to index content: {e}")
            raise
    
    def add_content_sync(self, blocks: List[ContentBlock]) -> None:
        """Synchronous version of add_content"""
        try:
            texts = []
            block_ids = []
            
            for block in blocks:
                self.content_blocks[block.id] = block
                self.modality_index[block.type].append(block.id)
                
                if block.type == ContentType.TEXT:
                    texts.append(block.content)
                else:
                    texts.append(f"{block.type.value}: {block.content[:500]}")
                
                block_ids.append(block.id)
            
            if texts:
                embeddings = self.embedding_provider.embed_text_sync(texts)
                
                for bid, embedding in zip(block_ids, embeddings):
                    self.embeddings[bid] = embedding
                    self.content_blocks[bid].embeddings = embedding.tolist()
                
                logger.info(f"Indexed {len(blocks)} content blocks")
        except Exception as e:
            logger.error(f"Failed to index content: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        modality_filter: Optional[ContentType] = None,
        min_score: float = 0.3,
        query_language: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant content with multilingual support
        
        Args:
            query: Search query
            top_k: Number of results to return
            modality_filter: Filter by content type
            min_score: Minimum similarity score
            query_language: Language of query for multilingual retrieval
        
        Returns:
            List of retrieval results ranked by multilingual similarity
        """
        try:
            from ..i18n import LanguageDetector
            
            # Detect query language if not provided
            if query_language is None:
                detector = LanguageDetector()
                query_language = detector.detect(query)
            
            # Embed query
            query_embedding = await self.embedding_provider.embed_text([query])
            query_embedding = query_embedding[0]
            
            # Vector similarity search with multilingual support
            results = self._vector_search(
                query_embedding, 
                top_k if not self.enable_rerank else top_k * 2,  # Get more results for reranking
                modality_filter, 
                min_score,
                query_language=query_language
            )
            
            # Rerank results if enabled
            if self.enable_rerank and self.reranker:
                results = self.reranker.rerank(query, results, top_k=top_k)
            
            return results[:top_k]  # Enforce top_k limit
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def retrieve_sync(
        self,
        query: str,
        top_k: int = 5,
        modality_filter: Optional[ContentType] = None,
        min_score: float = 0.3,
        query_language: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Synchronous retrieval with multilingual support
        
        Args:
            query: Search query
            top_k: Number of results to return
            modality_filter: Filter by content type
            min_score: Minimum similarity score
            query_language: Language of query for multilingual retrieval
        
        Returns:
            List of retrieval results ranked by multilingual similarity
        """
        try:
            from ..i18n import LanguageDetector
            
            # Detect query language if not provided
            if query_language is None:
                detector = LanguageDetector()
                query_language = detector.detect(query)
            
            query_embedding = self.embedding_provider.embed_text_sync([query])
            query_embedding = query_embedding[0]
            
            results = self._vector_search(
                query_embedding, 
                top_k if not self.enable_rerank else top_k * 2,  # Get more results for reranking
                modality_filter, 
                min_score,
                query_language=query_language
            )
            
            # Rerank results if enabled
            if self.enable_rerank and self.reranker:
                results = self.reranker.rerank(query, results, top_k=top_k)
            
            return results[:top_k]  # Enforce top_k limit
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        modality_filter: Optional[ContentType],
        min_score: float,
        query_language: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform vector similarity search with multilingual support
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            modality_filter: Filter by content type
            min_score: Minimum similarity score
            query_language: Query language for cross-lingual retrieval
        
        Returns:
            List of retrieval results ranked by multilingual similarity
        """
        from ..i18n import LanguageDetector, CrosslingualRetrieval
        
        scores = []
        detector = LanguageDetector()
        crosslingual = CrosslingualRetrieval()
        
        # Detect query language if not provided
        if query_language is None:
            # This would ideally be passed in, but we can detect from context
            query_language = "en"
        
        for block_id, embedding in self.embeddings.items():
            block = self.content_blocks.get(block_id)
            if not block:
                continue
            
            # Apply modality filter
            if modality_filter and block.type != modality_filter:
                continue
            
            # Compute base cosine similarity
            base_sim = self._cosine_similarity(query_embedding, embedding)
            
            # Apply multilingual adjustment
            doc_language = block.language if hasattr(block, 'language') else 'en'
            adjusted_sim = crosslingual.get_cross_lingual_score(
                query_language, 
                doc_language, 
                base_sim
            )
            
            # Apply same-language boost
            boost = crosslingual.get_same_language_boost(query_language, doc_language)
            final_sim = adjusted_sim * boost
            
            if final_sim >= min_score:
                scores.append((block_id, final_sim, block, doc_language))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for block_id, score, block, doc_language in scores[:top_k]:
            result = RetrievalResult(
                doc_id=block_id,
                score=float(score),
                content=block.content,
                content_type=block.type,
                metadata=block.metadata or {
                    "language": doc_language,
                    "query_language": query_language,
                    "cross_lingual": query_language != doc_language
                }
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "indexed_blocks": len(self.content_blocks),
            "modality_distribution": {
                ct.value: len(bid_list)
                for ct, bid_list in self.modality_index.items()
            },
        }

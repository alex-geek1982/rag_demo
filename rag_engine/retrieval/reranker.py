"""
RAG Engine - Reranker for result re-ranking
"""
from typing import List, Optional, Callable, Any, Dict
import logging
import numpy as np
from abc import ABC, abstractmethod

from ..types import RetrievalResult, ContentType


logger = logging.getLogger(__name__)


class RerankProvider(ABC):
    """Abstract base for reranking providers"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query string
            docs: List of retrieval results
            top_k: Optional number of top results to return
        
        Returns:
            Reranked list of retrieval results
        """
        pass


class SimpleReranker(RerankProvider):
    """Simple reranker using keyword matching and length considerations"""
    
    def __init__(self, query_weight: float = 0.6, length_weight: float = 0.2, score_weight: float = 0.2):
        """
        Initialize simple reranker
        
        Args:
            query_weight: Weight for query term overlap (0-1)
            length_weight: Weight for content length appropriateness (0-1)
            score_weight: Weight for original embedding score (0-1)
        """
        self.query_weight = query_weight
        self.length_weight = length_weight
        self.score_weight = score_weight
    
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank using simple heuristics"""
        if not docs:
            return docs
        
        query_terms = set(query.lower().split())
        reranked = []
        
        for doc in docs:
            # Calculate query term overlap
            doc_terms = set(doc.content[:500].lower().split())
            overlap = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            
            # Calculate content length score (balanced around 200-500 words)
            word_count = len(doc.content.split())
            length_score = 1.0 - abs(300 - word_count) / 1000.0
            length_score = max(0, min(1, length_score))
            
            # Combine scores
            rerank_score = (
                overlap * self.query_weight +
                length_score * self.length_weight +
                doc.score * self.score_weight
            )
            
            doc.score = rerank_score
            reranked.append(doc)
        
        # Sort by reranked score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k if specified
        if top_k:
            return reranked[:top_k]
        return reranked


class LLMReranker(RerankProvider):
    """Reranker using LLM for relevance assessment"""
    
    def __init__(self, llm_model_func: Optional[Callable] = None, batch_size: int = 10):
        """
        Initialize LLM reranker
        
        Args:
            llm_model_func: Function to call LLM for relevance scoring
            batch_size: Number of documents to score in parallel
        """
        self.llm_model_func = llm_model_func
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank using LLM relevance assessment"""
        if not docs or not self.llm_model_func:
            return docs[:top_k] if top_k else docs
        
        reranked = []
        
        # Process in batches
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i:i + self.batch_size]
            
            # Create batch prompt for LLM
            batch_prompt = self._create_batch_prompt(query, batch)
            
            try:
                # Get LLM scores
                response = self.llm_model_func(batch_prompt)
                scores = self._parse_scores(response, len(batch))
                
                # Update document scores
                for doc, score in zip(batch, scores):
                    doc.score = (doc.score + score) / 2.0  # Average with original score
                    reranked.append(doc)
            except Exception as e:
                logger.warning(f"LLM reranking failed: {e}, using original scores")
                reranked.extend(batch)
        
        # Sort by score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k if specified
        if top_k:
            return reranked[:top_k]
        return reranked
    
    def _create_batch_prompt(self, query: str, docs: List[RetrievalResult]) -> str:
        """Create batch prompt for LLM scoring"""
        prompt = f"""Rate the relevance of each document to the query on a scale of 0-1.
Query: {query}

Documents:
"""
        for i, doc in enumerate(docs, 1):
            content_preview = doc.content[:200].replace("\n", " ")
            prompt += f"{i}. {content_preview}...\n"
        
        prompt += "\nProvide relevance scores as comma-separated decimal values (e.g., 0.9,0.7,0.5)"
        
        return prompt
    
    @staticmethod
    def _parse_scores(response: str, num_docs: int) -> List[float]:
        """Parse LLM response to extract scores"""
        try:
            # Try to extract scores from response
            scores_str = response.split(":")[-1].strip()
            scores = [float(s.strip()) for s in scores_str.split(",")]
            
            # Pad with 0.5 if not enough scores
            while len(scores) < num_docs:
                scores.append(0.5)
            
            return scores[:num_docs]
        except Exception as e:
            logger.warning(f"Failed to parse LLM scores: {e}")
            return [0.5] * num_docs


class CrossEncoderReranker(RerankProvider):
    """Reranker using cross-encoder model for better relevance assessment"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: str = "cpu"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Cross-encoder model name from HuggingFace
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device, max_length=512)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
    
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank using cross-encoder model"""
        if not docs or not self.model:
            return docs[:top_k] if top_k else docs
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in docs:
                query_doc_pairs.append([query, doc.content[:512]])
            
            # Get scores from cross-encoder
            scores = self.model.predict(query_doc_pairs)
            
            # Update document scores (normalize to 0-1 range)
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            reranked = []
            for doc, score in zip(docs, normalized_scores):
                doc.score = float(score)
                reranked.append(doc)
            
            # Sort by score
            reranked.sort(key=lambda x: x.score, reverse=True)
            
            # Return top-k if specified
            if top_k:
                return reranked[:top_k]
            return reranked
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return docs[:top_k] if top_k else docs


class HybridReranker(RerankProvider):
    """Hybrid reranker combining multiple reranking strategies"""
    
    def __init__(
        self,
        primary_reranker: Optional[RerankProvider] = None,
        secondary_reranker: Optional[RerankProvider] = None,
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3
    ):
        """
        Initialize hybrid reranker
        
        Args:
            primary_reranker: Primary reranking strategy
            secondary_reranker: Secondary reranking strategy
            primary_weight: Weight for primary reranker (0-1)
            secondary_weight: Weight for secondary reranker (0-1)
        """
        self.primary_reranker = primary_reranker or SimpleReranker()
        self.secondary_reranker = secondary_reranker
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
    
    def rerank(
        self,
        query: str,
        docs: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid reranking"""
        if not docs:
            return docs
        
        # Keep original scores
        original_scores = {doc.doc_id: doc.score for doc in docs}
        
        # Primary reranking
        primary_reranked = self.primary_reranker.rerank(
            query, 
            [r for r in docs],  # Create copies to avoid modifying originals
            top_k=None
        )
        primary_scores = {doc.doc_id: doc.score for doc in primary_reranked}
        
        # Secondary reranking if available
        if self.secondary_reranker:
            secondary_reranked = self.secondary_reranker.rerank(
                query,
                [r for r in docs],  # Create copies
                top_k=None
            )
            secondary_scores = {doc.doc_id: doc.score for doc in secondary_reranked}
        else:
            secondary_scores = {}
        
        # Combine scores
        reranked = []
        for doc in docs:
            primary_score = primary_scores.get(doc.doc_id, 0.5)
            secondary_score = secondary_scores.get(doc.doc_id, 0.5) if secondary_scores else 0.5
            
            combined_score = (
                primary_score * self.primary_weight +
                secondary_score * self.secondary_weight
            )
            
            doc.score = combined_score
            reranked.append(doc)
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Return top-k if specified
        if top_k:
            return reranked[:top_k]
        return reranked

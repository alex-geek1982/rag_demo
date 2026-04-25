"""
BM25 Full-Text Retriever and Hybrid Score Fusion

Provides:
- BM25 full-text search with configurable parameters
- Multiple score normalization methods
- Multiple fusion strategies for combining retrieval results
- Proper handling of deduplication and result ranking
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..types import RetrievalResult, ContentType
from ..config import BM25Config, HybridRetrievalConfig

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Matching 25) full-text retriever.
    
    Uses the Okapi BM25 algorithm for ranking text documents.
    Supports custom tokenization and language-specific stemming.
    """
    
    def __init__(self, config: BM25Config, documents: Optional[List[str]] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            config: BM25 configuration
            documents: Optional list of documents to index
        """
        self.config = config
        self.k1 = config.k1  # Saturation parameter (typically 1.5)
        self.b = config.b    # Document length normalization (typically 0.75)
        self.min_token_length = config.min_token_length
        
        # Initialize tokenizer
        self._init_tokenizer(config.language)
        
        # Corpus statistics
        self.corpus = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.avg_doc_len = 0.0
        
        # Index documents if provided
        if documents:
            self.index(documents)
    
    def _init_tokenizer(self, language: str) -> None:
        """Initialize tokenizer with language-specific support."""
        try:
            import nltk
            from nltk.stem import SnowballStemmer
            
            # Ensure required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except (LookupError, RuntimeError, Exception):
                try:
                    nltk.download('punkt', quiet=True)
                except Exception as e:
                    logger.debug(f"Failed to download NLTK data: {e}")
            
            self.stemmer = SnowballStemmer(language if language != "english" else "english")
            self.use_stemming = True
        except (ImportError, Exception) as e:
            logger.debug(f"NLTK stemming not available: {e}, using simple tokenization")
            self.use_stemming = False
            self.stemmer = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with optional stemming.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace + punctuation-based tokenization
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        # Apply stemming if available
        if self.use_stemming and self.stemmer:
            try:
                tokens = [self.stemmer.stem(t) for t in tokens]
            except Exception as e:
                logger.debug(f"Stemming failed: {e}, using original tokens")
        
        return tokens
    
    def index(self, documents: List[str]) -> None:
        """
        Index documents using BM25.
        
        Args:
            documents: List of documents to index
        """
        self.corpus = documents
        self.doc_tokens = []
        self.doc_freqs = []
        
        # Tokenize all documents and compute term frequencies
        total_doc_len = 0
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_tokens.append(tokens)
            total_doc_len += len(tokens)
            
            # Count term frequencies
            freq_map: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freq_map[token] += 1
            self.doc_freqs.append(dict(freq_map))
        
        # Compute average document length
        self.avg_doc_len = total_doc_len / len(documents) if documents else 0.0
        
        # Compute IDF (Inverse Document Frequency)
        self._compute_idf()
        
        logger.info(f"BM25 indexed {len(documents)} documents with avg length {self.avg_doc_len:.2f}")
    
    def _compute_idf(self) -> None:
        """Compute IDF scores for all terms in the corpus."""
        num_docs = len(self.corpus)
        
        if num_docs == 0:
            return
        
        # Count document frequency for each term
        doc_freq: Dict[str, int] = defaultdict(int)
        for freq_map in self.doc_freqs:
            for term in freq_map.keys():
                doc_freq[term] += 1
        
        # Compute IDF with smoothing
        self.idf = {}
        for term, df in doc_freq.items():
            # IDF = log((N - df + 0.5) / (df + 0.5))
            # This is the BM25 variant of IDF
            self.idf[term] = np.log((num_docs - df + 0.5) / (df + 0.5))
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        if not self.corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Score all documents
        scores = []
        for doc_idx, (tokens, freq_map) in enumerate(zip(self.doc_tokens, self.doc_freqs)):
            score = self._compute_bm25_score(query_tokens, doc_idx, freq_map)
            scores.append((doc_idx, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _compute_bm25_score(self, query_tokens: List[str], doc_idx: int, 
                            freq_map: Dict[str, int]) -> float:
        """
        Compute BM25 score for a document.
        
        Args:
            query_tokens: Query tokens
            doc_idx: Document index
            freq_map: Term frequency map for the document
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = len(self.doc_tokens[doc_idx])
        
        for token in query_tokens:
            if token not in freq_map:
                continue
            
            # Get term frequency in document
            tf = freq_map[token]
            
            # Get IDF for term
            idf = self.idf.get(token, 0.0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return max(0.0, score)


class ScoreNormalizer:
    """
    Normalize and fuse scores from different retrieval methods.
    
    Supports multiple normalization strategies:
    - Min-Max: Normalize to [0, 1] range
    - Sigmoid: Smooth normalization
    - Rank: Convert to rank-based scores
    """
    
    @staticmethod
    def normalize_minmax(scores: List[float], eps: float = 1e-8) -> List[float]:
        """
        Normalize scores using Min-Max normalization.
        
        Args:
            scores: List of scores
            eps: Small value to avoid division by zero
            
        Returns:
            Normalized scores in [0, 1] range
        """
        if not scores or len(scores) == 0:
            return []
        
        scores_array = np.array(scores, dtype=float)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        # Handle case where all scores are the same
        if max_score - min_score < eps:
            return [0.5] * len(scores)
        
        # Min-Max normalization
        normalized = (scores_array - min_score) / (max_score - min_score + eps)
        return normalized.tolist()
    
    @staticmethod
    def normalize_sigmoid(scores: List[float], scale: float = 1.0) -> List[float]:
        """
        Normalize scores using Sigmoid function.
        
        Args:
            scores: List of scores
            scale: Scaling factor for sigmoid
            
        Returns:
            Normalized scores in approximately [0, 1] range
        """
        if not scores:
            return []
        
        scores_array = np.array(scores, dtype=float)
        # Sigmoid: 1 / (1 + e^(-x/scale))
        normalized = 1.0 / (1.0 + np.exp(-scores_array / scale))
        return normalized.tolist()
    
    @staticmethod
    def normalize_rank(results: List[Tuple[str, float]], k: float = 60.0) -> Dict[str, float]:
        """
        Normalize scores using Reciprocal Rank Fusion (RRF).
        
        Args:
            results: List of (doc_id, rank) tuples sorted by rank
            k: RRF parameter (typically 60)
            
        Returns:
            Dict mapping doc_id to RRF score
        """
        scores = {}
        for rank, (doc_id, _) in enumerate(results, 1):
            # RRF score: 1 / (k + rank)
            scores[doc_id] = 1.0 / (k + rank)
        return scores


class HybridFuser:
    """
    Fuse multiple retrieval results using configurable strategies.
    
    Handles:
    - Score normalization from different sources
    - Multiple fusion strategies (weighted average, RRF, etc.)
    - Deduplication and result merging
    - Final ranking
    """
    
    def __init__(self, config: HybridRetrievalConfig):
        """
        Initialize hybrid fuser.
        
        Args:
            config: Hybrid retrieval configuration
        """
        self.config = config
        self.normalizer = ScoreNormalizer()
    
    def fuse(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        graph_results: Optional[List[RetrievalResult]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Fuse results from multiple retrieval sources.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            graph_results: Optional results from graph search
            top_k: Number of final results to return
            
        Returns:
            Fused and ranked results
        """
        if graph_results is None:
            graph_results = []
        
        # Choose fusion strategy
        if self.config.fusion_strategy == "rrf":
            fused = self._fuse_rrf(vector_results, bm25_results, graph_results, top_k)
        elif self.config.fusion_strategy == "weighted_avg":
            fused = self._fuse_weighted_avg(vector_results, bm25_results, graph_results, top_k)
        elif self.config.fusion_strategy == "max":
            fused = self._fuse_max(vector_results, bm25_results, graph_results, top_k)
        elif self.config.fusion_strategy == "min":
            fused = self._fuse_min(vector_results, bm25_results, graph_results, top_k)
        else:
            logger.warning(f"Unknown fusion strategy: {self.config.fusion_strategy}, using weighted_avg")
            fused = self._fuse_weighted_avg(vector_results, bm25_results, graph_results, top_k)
        
        return sorted(fused, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _fuse_weighted_avg(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Fuse using weighted average of normalized scores.
        
        This is the most straightforward and interpretable approach.
        """
        # Normalize scores
        vector_scores = self.normalizer.normalize_minmax([r.score for r in vector_results])
        bm25_scores = self.normalizer.normalize_minmax([r.score for r in bm25_results])
        graph_scores = self.normalizer.normalize_minmax([r.score for r in graph_results]) if graph_results else []
        
        # Create score maps
        fused_scores: Dict[str, float] = {}
        source_channels: Dict[str, set] = defaultdict(set)
        result_map: Dict[str, RetrievalResult] = {}
        
        # Add vector results
        for result, norm_score in zip(vector_results, vector_scores):
            if result.doc_id not in fused_scores:
                fused_scores[result.doc_id] = 0.0
                result_map[result.doc_id] = result
            fused_scores[result.doc_id] += norm_score * self.config.vector_weight
            source_channels[result.doc_id].add("vector")
        
        # Add BM25 results
        for result, norm_score in zip(bm25_results, bm25_scores):
            if result.doc_id not in fused_scores:
                fused_scores[result.doc_id] = 0.0
                result_map[result.doc_id] = result
            else:
                # Update existing result with higher score if available
                if norm_score * self.config.bm25_weight > fused_scores[result.doc_id]:
                    result_map[result.doc_id].content = result.content
            fused_scores[result.doc_id] += norm_score * self.config.bm25_weight
            source_channels[result.doc_id].add("bm25")
        
        # Add graph results (optional)
        for result, norm_score in zip(graph_results, graph_scores):
            if result.doc_id not in fused_scores:
                fused_scores[result.doc_id] = 0.0
                result_map[result.doc_id] = result
            fused_scores[result.doc_id] += norm_score * self.config.graph_weight
            source_channels[result.doc_id].add("graph")
        
        # Create fused results
        fused = []
        for doc_id, score in fused_scores.items():
            result = result_map[doc_id]
            result.score = min(1.0, score)  # Cap at 1.0
            
            # Update retrieval channel metadata
            channels = sorted(list(source_channels[doc_id]))
            result.metadata["retrieval_channel"] = "+".join(channels)
            result.metadata["fusion_method"] = "weighted_avg"
            
            fused.append(result)
        
        return fused
    
    def _fuse_rrf(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Fuse using Reciprocal Rank Fusion (RRF).
        
        RRF is particularly effective when combining results from different
        retrieval methods as it treats ranking position equally important
        regardless of source.
        """
        # Convert to rank format
        vector_ranked = [(r.doc_id, i) for i, r in enumerate(vector_results, 1)]
        bm25_ranked = [(r.doc_id, i) for i, r in enumerate(bm25_results, 1)]
        graph_ranked = [(r.doc_id, i) for i, r in enumerate(graph_results, 1)] if graph_results else []
        
        # Compute RRF scores for each source
        vector_rrf = self.normalizer.normalize_rank(vector_ranked, self.config.rrf_k)
        bm25_rrf = self.normalizer.normalize_rank(bm25_ranked, self.config.rrf_k)
        graph_rrf = self.normalizer.normalize_rank(graph_ranked, self.config.rrf_k) if graph_ranked else {}
        
        # Fuse RRF scores with weights
        fused_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}
        source_channels: Dict[str, set] = defaultdict(set)
        
        # Add weighted RRF scores
        for doc_id, rrf_score in vector_rrf.items():
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += rrf_score * self.config.vector_weight
            source_channels[doc_id].add("vector")
        
        for doc_id, rrf_score in bm25_rrf.items():
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += rrf_score * self.config.bm25_weight
            source_channels[doc_id].add("bm25")
        
        for doc_id, rrf_score in graph_rrf.items():
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += rrf_score * self.config.graph_weight
            source_channels[doc_id].add("graph")
        
        # Map results
        for result in vector_results + bm25_results + graph_results:
            if result.doc_id not in result_map:
                result_map[result.doc_id] = result
        
        # Create fused results
        fused = []
        for doc_id, score in fused_scores.items():
            result = result_map[doc_id]
            result.score = score
            channels = sorted(list(source_channels[doc_id]))
            result.metadata["retrieval_channel"] = "+".join(channels)
            result.metadata["fusion_method"] = "rrf"
            fused.append(result)
        
        return fused
    
    def _fuse_max(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Fuse by taking maximum score across sources."""
        result_map: Dict[str, RetrievalResult] = {}
        max_scores: Dict[str, float] = {}
        source_channels: Dict[str, set] = defaultdict(set)
        
        for result in vector_results:
            if result.doc_id not in result_map:
                result_map[result.doc_id] = result
                max_scores[result.doc_id] = result.score
            else:
                max_scores[result.doc_id] = max(max_scores[result.doc_id], result.score)
            source_channels[result.doc_id].add("vector")
        
        for result in bm25_results:
            if result.doc_id not in result_map:
                result_map[result.doc_id] = result
                max_scores[result.doc_id] = result.score
            else:
                max_scores[result.doc_id] = max(max_scores[result.doc_id], result.score)
            source_channels[result.doc_id].add("bm25")
        
        for result in graph_results:
            if result.doc_id not in result_map:
                result_map[result.doc_id] = result
                max_scores[result.doc_id] = result.score
            else:
                max_scores[result.doc_id] = max(max_scores[result.doc_id], result.score)
            source_channels[result.doc_id].add("graph")
        
        fused = []
        for doc_id, score in max_scores.items():
            result = result_map[doc_id]
            result.score = score
            channels = sorted(list(source_channels[doc_id]))
            result.metadata["retrieval_channel"] = "+".join(channels)
            result.metadata["fusion_method"] = "max"
            fused.append(result)
        
        return fused
    
    def _fuse_min(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Fuse by taking minimum score across sources (conservative)."""
        all_results = vector_results + bm25_results + graph_results
        result_map: Dict[str, RetrievalResult] = {}
        min_scores: Dict[str, float] = {}
        source_channels: Dict[str, set] = defaultdict(set)
        
        for result in all_results:
            if result.doc_id not in result_map:
                result_map[result.doc_id] = result
                min_scores[result.doc_id] = result.score
            else:
                min_scores[result.doc_id] = min(min_scores[result.doc_id], result.score)
            
            # Determine source channel
            channel = result.metadata.get("retrieval_channel", "unknown")
            source_channels[result.doc_id].add(channel)
        
        fused = []
        for doc_id, score in min_scores.items():
            result = result_map[doc_id]
            result.score = score
            channels = sorted(list(source_channels[doc_id]))
            result.metadata["retrieval_channel"] = "+".join(channels)
            result.metadata["fusion_method"] = "min"
            fused.append(result)
        
        return fused

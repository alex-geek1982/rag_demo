"""
RAG Engine - Multilingual support for content retrieval and generation

This module handles true multilingual RAG capabilities:
- Multi-language document content processing
- Multi-language query understanding
- Cross-lingual retrieval
- Language detection and routing
- Multilingual embeddings
"""
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Language codes and metadata
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "embedding_model": "text-embedding-3-large"},
    "zh": {"name": "Chinese", "embedding_model": "text-embedding-3-large"},
    "ja": {"name": "Japanese", "embedding_model": "text-embedding-3-large"},
    "ko": {"name": "Korean", "embedding_model": "text-embedding-3-large"},
    "es": {"name": "Spanish", "embedding_model": "text-embedding-3-large"},
    "fr": {"name": "French", "embedding_model": "text-embedding-3-large"},
    "de": {"name": "German", "embedding_model": "text-embedding-3-large"},
}


class LanguageDetector:
    """Detect language of text content"""
    
    @staticmethod
    def detect(text: str) -> str:
        """
        Detect language of text using simple heuristics
        
        Args:
            text: Text to detect language for
        
        Returns:
            Language code (en, zh, ja, ko, es, fr, de)
        """
        if not text or len(text) < 1:
            return "en"
        
        # Sample first 100 chars for detection
        sample = text[:100]
        
        # Chinese characters (CJK Unified Ideographs)
        if any('\u4e00' <= c <= '\u9fff' for c in sample):
            return "zh"
        
        # Japanese Hiragana/Katakana
        if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in sample):
            return "ja"
        
        # Korean Hangul
        if any('\uac00' <= c <= '\ud7af' for c in sample):
            return "ko"
        
        # Spanish indicators
        if any(c in sample for c in ['ñ', '¿', '¡']) or 'que' in sample.lower():
            return "es"
        
        # French indicators
        if 'ç' in sample or 'œ' in sample or 'ù' in sample:
            return "fr"
        
        # German indicators
        if any(c in sample for c in ['ä', 'ö', 'ü', 'ß']):
            return "de"
        
        return "en"


class MultilingualEmbedding:
    """Handle multilingual embeddings with language awareness"""
    
    def __init__(self, embedding_provider=None):
        """
        Initialize multilingual embedding handler
        
        Args:
            embedding_provider: Embedding provider instance (e.g., OpenAIEmbedding)
        """
        self.embedding_provider = embedding_provider
        self.language_cache: Dict[str, str] = {}
    
    def embed_with_language_tag(self, text: str, language: Optional[str] = None) -> Tuple[List[float], str]:
        """
        Embed text with language detection/specification
        
        Args:
            text: Text to embed
            language: Optional language code override
        
        Returns:
            Tuple of (embedding_vector, detected_language)
        """
        if language is None:
            language = LanguageDetector.detect(text)
        
        # Create language-tagged text for better multilingual retrieval
        # This helps the embedding model understand context
        language_name = SUPPORTED_LANGUAGES.get(language, {}).get("name", "Unknown")
        tagged_text = f"[{language_name}] {text}"
        
        embedding = self.embedding_provider.embed_text(tagged_text) if self.embedding_provider else None
        
        return embedding, language
    
    def get_language(self, text: str) -> str:
        """Get detected language for text"""
        return LanguageDetector.detect(text)


class CrosslingualRetrieval:
    """Handle cross-lingual retrieval (search in one language, match in another)"""
    
    def __init__(self):
        """Initialize cross-lingual retrieval"""
        self.language_pairs: Dict[Tuple[str, str], float] = {
            # Language pairs with affinity scores (0-1)
            ("en", "zh"): 0.7,
            ("en", "ja"): 0.65,
            ("en", "ko"): 0.65,
            ("zh", "ja"): 0.6,
            ("zh", "ko"): 0.6,
            ("ja", "ko"): 0.6,
            ("es", "fr"): 0.75,
            ("es", "de"): 0.7,
            ("fr", "de"): 0.75,
        }
    
    def get_cross_lingual_score(self, query_language: str, document_language: str, 
                               base_similarity: float) -> float:
        """
        Adjust similarity score for cross-lingual retrieval
        
        Args:
            query_language: Language of query
            document_language: Language of document
            base_similarity: Base similarity score
        
        Returns:
            Adjusted similarity score
        """
        if query_language == document_language:
            return base_similarity
        
        # Apply cross-lingual penalty
        pair = (query_language, document_language) if (query_language, document_language) in self.language_pairs else (document_language, query_language)
        affinity = self.language_pairs.get(pair, 0.5)
        
        return base_similarity * affinity
    
    def get_same_language_boost(self, query_language: str, document_language: str) -> float:
        """
        Get similarity boost when query and document are same language
        
        Returns:
            Boost multiplier (>= 1.0)
        """
        if query_language == document_language:
            return 1.15  # 15% boost for same language matches
        return 1.0


class I18n:
    """
    True multilingual support for RAG system
    
    Handles:
    - Multilingual content processing
    - Language detection and routing
    - Cross-lingual retrieval
    - Multilingual embeddings
    - Language-specific optimizations
    """
    
    def __init__(self, embedding_provider=None):
        """
        Initialize multilingual support
        
        Args:
            embedding_provider: Embedding provider for language-aware embeddings
        """
        self.embedding_provider = embedding_provider
        self.detector = LanguageDetector()
        self.multilingual_embedding = MultilingualEmbedding(embedding_provider)
        self.crosslingual_retrieval = CrosslingualRetrieval()
        self.document_languages: Dict[str, str] = {}  # Track document languages
        self.supported_languages = list(SUPPORTED_LANGUAGES.keys())
    
    def detect_content_language(self, text: str) -> str:
        """
        Detect language of content
        
        Args:
            text: Text content to detect
        
        Returns:
            Language code
        """
        return self.detector.detect(text)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.supported_languages

# Global i18n instance
_i18n_instance: Optional[I18n] = None


def get_i18n(embedding_provider=None) -> I18n:
    """Get or create global i18n instance for multilingual RAG"""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = I18n(embedding_provider=embedding_provider)
    return _i18n_instance


def detect_language(text: str) -> str:
    """Detect language of text"""
    detector = LanguageDetector()
    return detector.detect(text)


def is_multilingual_content(texts: List[str]) -> bool:
    """
    Check if content contains multiple languages
    
    Args:
        texts: List of text segments
    
    Returns:
        True if multiple languages detected
    """
    if not texts:
        return False
    
    languages = set()
    detector = LanguageDetector()
    for text in texts:
        lang = detector.detect(text)
        languages.add(lang)
    
    return len(languages) > 1

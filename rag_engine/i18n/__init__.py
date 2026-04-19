"""
I18n module initialization - Multilingual RAG support
"""
from .i18n import (
    I18n,
    get_i18n,
    detect_language,
    is_multilingual_content,
    LanguageDetector,
    MultilingualEmbedding,
    CrosslingualRetrieval,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    "I18n",
    "get_i18n",
    "detect_language",
    "is_multilingual_content",
    "LanguageDetector",
    "MultilingualEmbedding",
    "CrosslingualRetrieval",
    "SUPPORTED_LANGUAGES",
]

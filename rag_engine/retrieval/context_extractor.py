"""
RAG Engine - Context extractor for multimodal content
"""
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging

from ..types import ContentBlock, Document, ContentType


logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context extraction"""
    context_window: int = 2  # Number of adjacent blocks to include
    context_mode: str = "hybrid"  # 'window', 'entity', 'hybrid'
    max_context_tokens: int = 1000
    include_headers: bool = True
    include_captions: bool = True
    filter_content_types: Optional[List[ContentType]] = None
    include_source_info: bool = True


class ContextExtractor:
    """Extract contextual information for content blocks"""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize context extractor
        
        Args:
            config: Context extraction configuration
        """
        self.config = config or ContextConfig()
        self.block_index: Dict[str, List[ContentBlock]] = {}  # doc_id -> blocks
        self.document_index: Dict[str, Document] = {}  # doc_id -> document
    
    def add_document(self, document: Document) -> None:
        """
        Register a document for context extraction
        
        Args:
            document: Document to index
        """
        self.document_index[document.id] = document
        self.block_index[document.id] = document.content_blocks
        logger.debug(f"Indexed document {document.id} with {len(document.content_blocks)} blocks")
    
    def extract_context(
        self,
        block: ContentBlock,
        doc_id: str,
        include_self: bool = True
    ) -> Dict[str, Any]:
        """
        Extract context for a content block
        
        Args:
            block: Content block to get context for
            doc_id: Document ID containing the block
            include_self: Whether to include the block itself
        
        Returns:
            Dictionary with context information
        """
        context = {
            "main_content": block if include_self else None,
            "surrounding_blocks": [],
            "source_document": None,
            "metadata": {}
        }
        
        # Get document info
        if doc_id in self.document_index:
            doc = self.document_index[doc_id]
            context["source_document"] = {
                "id": doc.id,
                "title": doc.title,
                "source_path": doc.source_path,
            }
        
        # Extract surrounding blocks based on mode
        if self.config.context_mode in ["window", "hybrid"]:
            surrounding = self._get_window_context(block, doc_id)
            context["surrounding_blocks"].extend(surrounding)
        
        # Add metadata
        context["metadata"] = {
            "context_window": self.config.context_window,
            "context_mode": self.config.context_mode,
            "total_context_blocks": len(context["surrounding_blocks"]) + (1 if include_self else 0),
            "includes_headers": self.config.include_headers,
            "includes_captions": self.config.include_captions,
        }
        
        return context
    
    def _get_window_context(
        self,
        block: ContentBlock,
        doc_id: str
    ) -> List[ContentBlock]:
        """Get context blocks within a window around the given block"""
        blocks = self.block_index.get(doc_id, [])
        if not blocks:
            return []
        
        try:
            block_idx = blocks.index(block)
        except ValueError:
            return []
        
        # Get surrounding blocks
        start_idx = max(0, block_idx - self.config.context_window)
        end_idx = min(len(blocks), block_idx + self.config.context_window + 1)
        
        surrounding = []
        for i in range(start_idx, end_idx):
            if i != block_idx:  # Exclude the main block
                b = blocks[i]
                # Apply filters
                if self.config.filter_content_types and b.type not in self.config.filter_content_types:
                    continue
                surrounding.append(b)
        
        return surrounding
    
    def get_context_string(
        self,
        block: ContentBlock,
        doc_id: str
    ) -> str:
        """
        Get context as a formatted string for LLM input
        
        Args:
            block: Main content block
            doc_id: Document ID
        
        Returns:
            Formatted context string
        """
        context = self.extract_context(block, doc_id, include_self=True)
        
        parts = []
        
        # Add source info
        if self.config.include_source_info and context["source_document"]:
            doc_info = context["source_document"]
            parts.append(f"Source: {doc_info['title']} (ID: {doc_info['id']})")
            parts.append("-" * 50)
        
        # Add main content
        if context["main_content"]:
            parts.append("Main Content:")
            parts.append(context["main_content"].content)
        
        # Add surrounding context
        if context["surrounding_blocks"]:
            parts.append("\nRelated Context:")
            for b in context["surrounding_blocks"]:
                parts.append(f"[{b.type.value}] {b.content[:200]}...")
        
        return "\n".join(parts)
    
    def format_context_for_multimodal(
        self,
        block: ContentBlock,
        doc_id: str,
        modality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format context specifically for multimodal processing
        
        Args:
            block: Content block  to format context for
            doc_id: Document ID
            modality: Optional specific modality (image, table, equation, etc.)
        
        Returns:
            Formatted context for multimodal processor
        """
        context = self.extract_context(block, doc_id, include_self=True)
        
        formatted = {
            "main_block": {
                "type": block.type.value if block.type else "unknown",
                "content": block.content,
                "metadata": block.metadata or {},
            },
            "surrounding_context": [],
            "document_context": context["source_document"],
            "extraction_config": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
            }
        }
        
        # Add surrounding blocks
        for b in context["surrounding_blocks"]:
            formatted["surrounding_context"].append({
                "type": b.type.value if b.type else "unknown",
                "content": b.content[:500],  # Limit size
                "source_file": b.source_file,
                "page_num": b.page_num,
            })
        
        return formatted


class ContextCache:
    """Cache for extracted contexts to improve performance"""
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize context cache
        
        Args:
            max_cache_size: Maximum number of cached contexts
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.access_count = 0
    
    def get(self, block_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached context"""
        key = f"{doc_id}:{block_id}"
        if key in self.cache:
            self.access_count += 1
            return self.cache[key]
        return None
    
    def set(self, block_id: str, doc_id: str, context: Dict[str, Any]) -> None:
        """Set cached context"""
        key = f"{doc_id}:{block_id}"
        
        # Simple eviction policy: remove oldest entry if cache full
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = context
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_items": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "access_count": self.access_count,
            "hit_rate": self.access_count / max(1, len(self.cache)),
        }

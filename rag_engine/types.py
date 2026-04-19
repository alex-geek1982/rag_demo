"""
RAG Engine - Base types and utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import json

_chunk_id_counter = count(1)


@dataclass
class Chunk:
    """Represents a single chunk of content"""
    text: str
    chunk_type: str  # 'title', 'text', 'table', 'image'
    id: str = field(default_factory=lambda: f"chunk-{next(_chunk_id_counter)}")
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    title: str = ""  # For hierarchical context
    title_level: int = 0  # Heading level (1-6)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "chunk_type": self.chunk_type,
            "id": self.id,
            "source_block_ids": self.source_block_ids,
            "metadata": self.metadata,
            "title": self.title,
            "title_level": self.title_level,
        }


class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    CHART = "chart"
    DIAGRAM = "diagram"


class ModalityType(str, Enum):
    """Modality type enumeration"""
    TEXT = "text"
    VISUAL = "visual"
    STRUCTURED = "structured"
    MATHEMATICAL = "mathematical"


@dataclass
class ContentBlock:
    """Single content block representation"""
    id: str
    type: ContentType
    content: str
    modality: ModalityType
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    page_num: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    language: str = "en"
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "modality": self.modality.value,
            "metadata": self.metadata,
            "source_file": self.source_file,
            "page_num": self.page_num,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
        }


@dataclass
class Document:
    """Document representation"""
    id: str
    title: str
    source_path: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    language: str = "en"
    _content_blocks: List[ContentBlock] = field(default_factory=list, init=False, repr=False)
    
    def add_content_block(self, block: ContentBlock) -> None:
        """Add content block to document (internal use for parsers)"""
        block.source_file = self.source_path
        self._content_blocks.append(block)
    
    def get_content_blocks(self) -> List[ContentBlock]:
        """Get content blocks (internal use for document processing)"""
        return self._content_blocks
    
    @property
    def content_blocks(self) -> List[ContentBlock]:
        """Access content blocks (for backward compatibility)
        
        Note: content_blocks are internal to Document and created by parsers.
        For knowledge graph building, use chunks_to_content_blocks() to convert chunks.
        """
        return self._content_blocks
    
    @content_blocks.setter
    def content_blocks(self, value: List[ContentBlock]) -> None:
        """Set content blocks (for backward compatibility)"""
        self._content_blocks = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "source_path": self.source_path,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "language": self.language,
        }


@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    modality: Optional[ModalityType] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_blocks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "modality": self.modality.value if self.modality else None,
            "attributes": self.attributes,
            "source_blocks": self.source_blocks,
        }


@dataclass
class Relationship:
    """Knowledge graph relationship"""
    id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Document retrieval result"""
    doc_id: str
    score: float
    content: str
    content_type: ContentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "content": self.content,
            "content_type": self.content_type.value,
            "metadata": self.metadata,
        }


@dataclass
class QueryResult:
    """Final query result with multilingual support"""
    query: str
    answer: str
    retrieved_docs: List[RetrievalResult] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None
    multimodal_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # For multilingual info: query_language, retrieved_languages, cross_lingual_retrieval
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_docs": [doc.to_dict() for doc in self.retrieved_docs],
            "sources": self.sources,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "multimodal_analysis": self.multimodal_analysis,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# ========== Helper Functions for Chunk to ContentBlock Conversion ==========

def chunks_to_content_blocks(chunks: List[Chunk], source_file: str = "", language: str = "en") -> List[ContentBlock]:
    """
    Convert chunks to content blocks for knowledge graph building.
    
    This conversion function should be called before building the knowledge graph:
    1. Document is processed and creates chunks
    2. chunks_to_content_blocks() is called to convert for KG building
    3. KnowledgeGraphBuilder processes the content blocks
    
    Args:
        chunks: List of Chunk objects
        source_file: Optional source file path
        language: Optional language code
        
    Returns:
        List of ContentBlock objects suitable for knowledge graph extraction
    """
    content_blocks = []
    
    for chunk in chunks:
        # Map chunk_type to ContentType
        type_mapping = {
            'title': ContentType.TEXT,
            'text': ContentType.TEXT,
            'table': ContentType.TABLE,
            'image': ContentType.IMAGE,
            'equation': ContentType.EQUATION,
            'code': ContentType.CODE,
        }
        
        content_type = type_mapping.get(chunk.chunk_type, ContentType.TEXT)
        
        # Map content type to modality
        modality_mapping = {
            ContentType.TEXT: ModalityType.TEXT,
            ContentType.TABLE: ModalityType.STRUCTURED,
            ContentType.IMAGE: ModalityType.VISUAL,
            ContentType.EQUATION: ModalityType.MATHEMATICAL,
            ContentType.CODE: ModalityType.TEXT,
            ContentType.CHART: ModalityType.VISUAL,
            ContentType.DIAGRAM: ModalityType.VISUAL,
        }
        
        modality = modality_mapping.get(content_type, ModalityType.TEXT)
        
        block = ContentBlock(
            id=chunk.id,
            type=content_type,
            content=chunk.text,
            modality=modality,
            metadata=chunk.metadata,
            source_file=source_file,
            language=language,
        )
        content_blocks.append(block)
    
    return content_blocks

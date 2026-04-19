"""
RAG Engine - Multimodal content processors
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging
import base64
from ..types import ContentBlock, ContentType, ModalityType, Entity, Relationship
from ..i18n import get_i18n


logger = logging.getLogger(__name__)


class BaseModalProcessor(ABC):
    """Base class for modal processors"""
    
    def __init__(self, language: str = "en"):
        """Initialize processor"""
        self.language = language
        self.i18n = get_i18n()
    
    @abstractmethod
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """
        Process content block
        
        Args:
            content_block: Content block to process
        
        Returns:
            Tuple of (processed_description, entity)
        """
        pass
    
    @abstractmethod
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports content type"""
        pass


class TextProcessor(BaseModalProcessor):
    """Text content processor"""
    
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports text"""
        return content_type == ContentType.TEXT
    
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """Process text content"""
        try:
            # Extract key sentences and summary
            text = content_block.content
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Create entity from key text
            entity = None
            if len(sentences) > 0:
                entity = Entity(
                    id=content_block.id,
                    name=f"Text Block {content_block.id}",
                    entity_type="TextContent",
                    description=sentences[0][:200] if sentences else "",
                    modality=ModalityType.TEXT,
                    attributes={
                        "sentence_count": len(sentences),
                        "char_count": len(text),
                        "language": self.language
                    },
                    source_blocks=[content_block.id]
                )
            
            return text, entity
        except Exception as e:
            logger.error("f""Processing failed: {str(e)}""")
            raise


class ImageProcessor(BaseModalProcessor):
    """Image content processor"""
    
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports images"""
        return content_type == ContentType.IMAGE
    
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """Process image content"""
        try:
            from PIL import Image
            
            image_path = content_block.content
            path = Path(image_path)
            
            if not path.exists():
                logger.warning(f"Image file not found: {image_path}")
                description = f"Reference to image: {path.name}"
            else:
                # Open and analyze image
                img = Image.open(path)
                
                # Generate image description
                description = f"Image: {path.name} (Format: {img.format}, Size: {img.width}x{img.height})"
                
                # Extract metadata
                metadata = img.info.copy()
                metadata.update({
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                })
            
            # Create entity
            entity = Entity(
                id=content_block.id,
                name=f"Image: {path.name}",
                entity_type="ImageContent",
                description=description,
                modality=ModalityType.VISUAL,
                attributes=content_block.metadata.copy() if content_block.metadata else {},
                source_blocks=[content_block.id]
            )
            
            return description, entity
        except Exception as e:
            logger.error("f""Processing failed: {str(e)}""")
            raise
    
    def get_image_base64(self, image_path: str) -> Optional[str]:
        """
        Convert image to base64
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image
        """
        try:
            path = Path(image_path)
            if path.exists():
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None


class TableProcessor(BaseModalProcessor):
    """Table/Structured data processor"""
    
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports tables"""
        return content_type == ContentType.TABLE
    
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """Process table content"""
        try:
            import pandas as pd
            from io import StringIO
            
            table_content = content_block.content
            
            # Try to parse as markdown table
            description = f"Table with structured data:\n{table_content[:500]}"
            
            # Extract statistics
            stats = self._extract_table_stats(table_content)
            
            entity = Entity(
                id=content_block.id,
                name=f"Table {content_block.id}",
                entity_type="TableContent",
                description=description,
                modality=ModalityType.STRUCTURED,
                attributes={
                    **stats,
                    "language": self.language,
                    **(content_block.metadata or {})
                },
                source_blocks=[content_block.id]
            )
            
            return description, entity
        except Exception as e:
            logger.error("f""Processing failed: {str(e)}""")
            raise
    
    def _extract_table_stats(self, table_content: str) -> Dict[str, Any]:
        """Extract statistics from table"""
        lines = table_content.strip().split('\n')
        rows = len([l for l in lines if l.strip()])
        
        # Count columns (by | character)
        cols = 0
        if rows > 0:
            cols = len([c for c in lines[0].split('|') if c.strip()])
        
        return {
            "row_count": rows,
            "column_count": cols,
        }


class EquationProcessor(BaseModalProcessor):
    """Mathematical equation processor"""
    
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports equations"""
        return content_type == ContentType.EQUATION
    
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """Process equation content"""
        try:
            equation = content_block.content
            
            # Generate description
            description = f"Mathematical equation: {equation[:100]}"
            
            entity = Entity(
                id=content_block.id,
                name=f"Equation {content_block.id}",
                entity_type="EquationContent",
                description=description,
                modality=ModalityType.MATHEMATICAL,
                attributes={
                    "latex": equation,
                    "length": len(equation),
                    "language": self.language,
                    **(content_block.metadata or {})
                },
                source_blocks=[content_block.id]
            )
            
            return description, entity
        except Exception as e:
            logger.error("f""Processing failed: {str(e)}""")
            raise


class CodeProcessor(BaseModalProcessor):
    """Code content processor"""
    
    def supports(self, content_type: ContentType) -> bool:
        """Check if processor supports code"""
        return content_type == ContentType.CODE
    
    def process(self, content_block: ContentBlock) -> Tuple[str, Optional[Entity]]:
        """Process code content"""
        try:
            code = content_block.content
            
            # Extract code language and description
            lang = content_block.metadata.get("language", "unknown") if content_block.metadata else "unknown"
            lines = code.split('\n')
            line_count = len([l for l in lines if l.strip()])
            
            description = f"Code block ({lang}, {line_count} lines):\n{code[:200]}"
            
            entity = Entity(
                id=content_block.id,
                name=f"Code {content_block.id}",
                entity_type="CodeContent",
                description=description,
                modality=ModalityType.STRUCTURED,
                attributes={
                    "language": lang,
                    "line_count": line_count,
                    **(content_block.metadata or {})
                },
                source_blocks=[content_block.id]
            )
            
            return description, entity
        except Exception as e:
            logger.error("f""Processing failed: {str(e)}""")
            raise


class ProcessorFactory:
    """Factory for creating modal processors"""
    
    PROCESSORS = [
        TextProcessor,
        ImageProcessor,
        TableProcessor,
        EquationProcessor,
        CodeProcessor,
    ]
    
    @classmethod
    def get_processor(cls, content_type: ContentType, language: str = "en") -> Optional[BaseModalProcessor]:
        """
        Get appropriate processor for content type
        
        Args:
            content_type: Content type
            language: Processing language
        
        Returns:
            Appropriate processor or None
        """
        for processor_class in cls.PROCESSORS:
            processor = processor_class(language)
            if processor.supports(content_type):
                return processor
        return None


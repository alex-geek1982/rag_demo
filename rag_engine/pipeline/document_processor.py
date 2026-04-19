"""
Document Processor - Parse and chunk documents
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from ..config import RAGEngineConfig, PDFProcessingConfig
from ..parsers import ParserFactory
from ..processors import ProcessorFactory
from ..types import Document, ContentBlock, ContentType, ModalityType, ModalityType
from .chunker import TitleChunker, TokenChunker, Chunk

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Responsibility: Parse documents and create content blocks.
    
    This class handles:
    - Document parsing (PDF, txt, md, etc.)
    - Content block extraction
    - Adaptive chunking based on configuration strategy
    
    It is independent of retrieval, indexing, or storage.
    """

    def __init__(self, config: RAGEngineConfig):
        """
        Initialize document processor.
        
        Args:
            config: RAG engine configuration
        """
        self.config = config
        
        # Initialize chunker based on configuration
        # chunker_type: 'title' (default) or 'token'
        chunker_type = config.processing.chunker_type.lower()
        
        if chunker_type == 'token':
            self.chunker = TokenChunker(
                chunk_token_size=config.processing.chunk_token_size,
                overlapped_percent=config.processing.chunk_overlap / 100.0,
                delimiters=['\n', '。', '！', '？', '.', '!', '?'],
                table_context_size=config.processing.chunk_size // 4,
                image_context_size=config.processing.chunk_size // 4
            )
        else:
            # Default: TitleChunker for structured documents
            self.chunker = TitleChunker(
                include_heading_content=False,
                use_outline=True
            )

    def process_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
        language: Optional[str] = None,
        markdown_path: Optional[str] = None,
    ) -> Document:
        """
        Process a single document into chunks.
        
        This method:
        1. Parses the document file
        2. Creates chunks from the parsed content
        3. Returns Document with chunks (for later KG building)
        
        Content blocks are NOT created here. They should be created when needed
        using chunks_to_content_blocks() helper before building the knowledge graph.
        
        Args:
            file_path: Path to document file
            doc_id: Optional document ID (auto-generated if not provided)
            doc_title: Optional document title
            language: Optional document language
            markdown_path: Optional path to pre-processed markdown file
            
        Returns:
            Document with chunks (ready for knowledge graph building)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate IDs if not provided
        doc_id = doc_id or f"doc_{path.stem}_{int(datetime.now().timestamp())}"
        doc_title = doc_title or path.stem
        language = language or self.config.language.default_language

        logger.info(f"Processing document: {doc_title}")

        try:
            # Parse document
            parser = ParserFactory.get_parser(
                str(path),
                doc_id,
                doc_title,
                language,
                pdf_config=self.config.pdf_processing,
                vision_config=self.config.vision,
            )
            
            if markdown_path is None:
                document = parser.parse(str(path))
                content_blocks_from_parser = document.get_content_blocks()
                if content_blocks_from_parser:
                    # If parser extracted content blocks, use them
                    temp_content_blocks = content_blocks_from_parser
                else:
                    # Otherwise create markdown block
                    temp_content_blocks = [ContentBlock(
                        id=f"{doc_id}_markdown",
                        type=ContentType.TEXT,
                        content=self._convert_to_markdown(document),
                        modality=ModalityType.TEXT,
                        metadata={"source": "converted_markdown"},
                        source_file=str(path),
                        page_num=None,
                        language=language,
                    )]
            else:
                # Load from pre-processed markdown
                document = Document(
                    id=doc_id,
                    title=doc_title,
                    source_path=str(path),
                    language=language,
                )
                if not Path(markdown_path).exists():
                    raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
                
                temp_content_blocks = [ContentBlock(
                    id=f"{doc_id}_markdown",
                    type=ContentType.TEXT,
                    content=Path(markdown_path).read_text(encoding='utf-8'),
                    modality=ModalityType.TEXT,
                    metadata={"source": "converted_markdown"},
                    source_file=str(path),
                    page_num=None,
                    language=language,
                )]
            
            # Create chunks from content blocks
            chunks = self.chunker.chunk(temp_content_blocks)
            
            # Merge chunks by token size (max 4096 tokens per chunk)
            merged_chunks = self._merge_chunks_by_token_size(chunks, max_tokens=4096)
            
            document.chunks = merged_chunks

            logger.info(
                f"Document processed: {doc_title} with {len(merged_chunks)} chunks"
            )
            return document

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    def process_folder(
        self,
        folder_path: str,
        language: Optional[str] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Process multiple documents in a folder.
        
        Args:
            folder_path: Path to folder
            language: Document language
            recursive: Whether to process subfolders
            
        Returns:
            List of Documents for each processed file
        """
        path = Path(folder_path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {folder_path}")

        results = []
        supported_extensions = (
            ".pdf",
            ".txt",
            ".md",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".jpg",
            ".jpeg",
            ".png",
        )

        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} documents to process")

        for file_path in files:
            try:
                doc = self.process_document(str(file_path), language=language)
                results.append(doc)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        return results

    def _process_content_blocks(self, document: Document) -> None:
        """Process content blocks with appropriate processors."""
        for block in document.content_blocks:
            # Skip processing markdown content blocks - they are already in final format
            if block.metadata and block.metadata.get("source") == "converted_markdown":
                continue
            
            processor = ProcessorFactory.get_processor(
                block.type, self.config.language.default_language
            )
            if processor:
                try:
                    description, entity = processor.process(block)
                    block.content = description  # Update with processed description
                except Exception as e:
                    logger.warning(f"Failed to process block {block.id}: {e}")

    def _sort_content_blocks_for_markdown(self, content_blocks: List[ContentBlock]) -> List[ContentBlock]:
        """Sort blocks by page and visual position so markdown preserves original layout order."""
        indexed_blocks = list(enumerate(content_blocks))

        def sort_key(item):
            index, block = item
            metadata = block.metadata or {}
            position = metadata.get("position") or {}
            page_num = block.page_num if block.page_num is not None else 0
            y0 = position.get("y0")
            y1 = position.get("y1")
            top = y0 if y0 is not None else (y1 if y1 is not None else float("inf"))
            x0 = position.get("x0", 0.0)
            return (page_num, top, x0, index)

        sorted_blocks = sorted(indexed_blocks, key=sort_key)
        return [block for _, block in sorted_blocks]

    def _normalize_text_for_markdown(self, text: str) -> str:
        """Normalize extracted text for Markdown output."""
        if text is None:
            return ""

        replacements = {
            "\uf0b2": "•",
            "\uf06c": "•",
            "\\uf06c": "•",
            "\u2022": "•",
            "\u2023": "•",
            "\u25e6": "•",
            "\u2043": "•",
            "\u2219": "•",
            "\uf0fc": "•",
            "\uf075": "•",
        }

        normalized = text
        for src, target in replacements.items():
            normalized = normalized.replace(src, target)

        # Normalize whitespace and remove zero-width control chars
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
        return normalized

    def _convert_to_markdown(self, document: Document) -> str:
        """Convert document content blocks to markdown format preserving hierarchy."""
        # First, analyze font size distribution to determine heading levels
        font_size_mapping = self._analyze_font_size_distribution(document)
        
        markdown_lines = []
        
        for block in self._sort_content_blocks_for_markdown(document.content_blocks):
            if block.type == ContentType.TEXT:
                # Check for heading level in metadata
                metadata = block.metadata or {}
                heading_level = None
                
                # Try different ways to get heading level
                if 'heading_level' in metadata:
                    heading_level = metadata['heading_level']
                elif 'layout_type' in metadata:
                    layout_type = metadata['layout_type']
                    if 'title' in layout_type.lower() or 'heading' in layout_type.lower():
                        try:
                            # Extract number from "title 1", "heading 2", etc.
                            parts = layout_type.split()
                            if len(parts) >= 2:
                                heading_level = int(parts[-1])
                        except (ValueError, IndexError):
                            pass
                elif metadata.get('block_type') == 'heading':
                    # Default to level 2 if marked as heading but no level
                    heading_level = 2
                
                # Use font_size to determine heading level if available
                if not heading_level and 'font_size' in metadata:
                    font_size = metadata['font_size']
                    heading_level = font_size_mapping.get(round(font_size, 1))
                
                content = self._normalize_text_for_markdown(block.content.strip())
                if heading_level and 1 <= heading_level <= 6:
                    markdown_lines.append('#' * heading_level + ' ' + content)
                else:
                    markdown_lines.append(content)
                
            elif block.type == ContentType.IMAGE:
                description = block.metadata.get('description', 'Image') if block.metadata else 'Image'
                description = self._normalize_text_for_markdown(description)
                image_path = block.content
                markdown_lines.append(f'![{description}]({image_path})')
                
            elif block.type == ContentType.TABLE:
                # Try to convert table content to markdown table
                table_md = self._convert_table_to_markdown(block.content)
                markdown_lines.append(table_md)
                
            elif block.type == ContentType.CODE:
                markdown_lines.append(f'```\n{block.content}\n```')
                
            elif block.type == ContentType.EQUATION:
                markdown_lines.append(f'$${block.content}$$')
                
            else:
                # Default: just add content
                markdown_lines.append(block.content.strip())
            
            # Add blank line between blocks for separation
            markdown_lines.append('')
        
        return '\n'.join(markdown_lines).strip()

    def _analyze_font_size_distribution(self, document: Document) -> Dict[float, int]:
        """
        Analyze font size distribution across the document to determine heading levels.
        
        Returns a mapping from font_size to heading_level based on document layout.
        """
        # Collect all font sizes from text blocks
        font_sizes = []
        for block in document.content_blocks:
            if block.type == ContentType.TEXT and block.metadata:
                font_size = block.metadata.get('font_size')
                if font_size is not None:
                    font_sizes.append(round(font_size, 1))
        
        if not font_sizes:
            return {}
        
        # Count frequency of each font size
        from collections import Counter
        size_counts = Counter(font_sizes)
        
        # Sort font sizes by size (descending) and then by frequency (ascending)
        # This prioritizes larger fonts and breaks ties by preferring less common sizes
        sorted_sizes = sorted(size_counts.keys(), key=lambda x: (-x, size_counts[x]))
        
        # Identify heading font sizes
        # Heuristics:
        # 1. Font sizes larger than the most common body text size
        # 2. Font sizes that appear relatively infrequently (likely headings)
        # 3. At least 2-3 distinct heading levels
        
        # Find the most common font size (likely body text)
        most_common_size = size_counts.most_common(1)[0][0]
        
        # Consider font sizes larger than body text as potential headings
        potential_headings = [size for size in sorted_sizes if size > most_common_size]
        
        # If we don't have enough large fonts, also consider fonts that are less common
        if len(potential_headings) < 3:
            # Add fonts that appear less frequently than the most common
            less_common = [size for size in sorted_sizes 
                         if size_counts[size] < size_counts[most_common_size] 
                         and size not in potential_headings]
            potential_headings.extend(less_common[:3])  # Add up to 3 more
        
        # Sort potential headings by size (largest first)
        potential_headings.sort(reverse=True)
        
        # Assign heading levels (1-6)
        font_size_mapping = {}
        for i, font_size in enumerate(potential_headings[:6]):  # Limit to 6 levels
            font_size_mapping[font_size] = i + 1
        
        logger.debug(f"Font size analysis: body_text={most_common_size}, headings={font_size_mapping}")
        return font_size_mapping

    def _convert_table_to_markdown(self, table_content: str) -> str:
        """Convert table content to markdown table format."""
        # Table content is already formatted as " | " separated rows from PDF extraction
        try:
            lines = table_content.strip().split('\n')
            if not lines:
                return table_content
            
            # Clean and process rows, removing duplicates
            rows = []
            seen = set()
            for line in lines:
                cells = None
                # Split by " | " separator used in PDF extraction
                if ' | ' in line:
                    cells = [cell.strip() for cell in line.split(' | ')]
                elif '\t' in line:
                    cells = [cell.strip() for cell in line.split('\t')]
                elif ',' in line:
                    cells = [cell.strip() for cell in line.split(',')]
                
                if cells:
                    row_tuple = tuple(cells)
                    if row_tuple not in seen:
                        rows.append(cells)
                        seen.add(row_tuple)
            
            if not rows:
                return table_content
            
            # Create markdown table
            markdown_table = []
            
            # Header row
            if len(rows) > 0:
                markdown_table.append('| ' + ' | '.join(rows[0]) + ' |')
                markdown_table.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
            
            # Data rows
            for row in rows[1:]:
                if len(row) == len(rows[0]):  # Ensure same number of columns
                    markdown_table.append('| ' + ' | '.join(row) + ' |')
                else:
                    # Handle rows with different column counts by padding
                    max_cols = len(rows[0])
                    while len(row) < max_cols:
                        row.append('')
                    markdown_table.append('| ' + ' | '.join(row[:max_cols]) + ' |')
            
            return '\n'.join(markdown_table)
            
        except Exception as e:
            logger.warning(f"Failed to convert table to markdown: {e}")
            return table_content

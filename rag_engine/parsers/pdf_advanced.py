"""
RAG Engine - Advanced PDF processor with layout understanding
Extracts images, text, and their relationships from PDFs
"""
import logging
import re
import os
import math
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import base64
import io

from ..types import Document, ContentBlock, ContentType, ModalityType


logger = logging.getLogger(__name__)


@dataclass
class ImageLocation:
    """Information about image location in PDF"""
    image_path: str
    page_num: int
    x0: float  # Left coordinate
    y0: float  # Top coordinate
    x1: float  # Right coordinate
    y1: float  # Bottom coordinate
    width: float
    height: float
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box as (x0, y0, x1, y1)"""
        return (self.x0, self.y0, self.x1, self.y1)
    
    def get_area(self) -> float:
        """Get image area"""
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class TextBlock:
    """Information about text block in PDF"""
    text: str
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    block_type: str  # 'text', 'table', etc.
    font_size: Optional[float] = None  # Font size if available
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box"""
        return (self.x0, self.y0, self.x1, self.y1)
    
    def distance_to(self, other: 'TextBlock') -> float:
        """Calculate distance to another block (center to center)"""
        self_center = ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
        other_center = ((other.x0 + other.x1) / 2, (other.y0 + other.y1) / 2)
        dx = self_center[0] - other_center[0]
        dy = self_center[1] - other_center[1]
        return (dx**2 + dy**2)**0.5


class LayoutElement(Enum):
    """Types of layout elements in PDF"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    HEADING = "heading"
    FORMULA = "formula"


class AdvancedPDFProcessor:
    """Advanced PDF processor with layout understanding"""
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        extract_text: bool = True,
        use_vision_api: bool = True,
        vision_api_key: Optional[str] = None,
        vision_base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        vision_provider: str = "openai",
        context_window_pixels: int = 200,
        min_image_area: int = 1000,
        max_surrounding_text_chars: int = 2000,
        filter_header_footer: bool = True,
        header_margin_ratio: float = 0.08,
        footer_margin_ratio: float = 0.08,
        header_footer_min_repeat_pages: int = 2,
        max_header_footer_text_length: int = 160,
    ):
        """
        Initialize advanced PDF processor
        
        Args:
            extract_images: Whether to extract and describe images
            extract_tables: Whether to extract tables as structured content
            extract_text: Whether to extract text
            use_vision_api: Whether to use Vision API for image description
            vision_api_key: API key for Vision service (defaults to OPENAI_API_KEY)
            vision_provider: Vision API provider ("openai" or "gemini")
            context_window_pixels: Pixel distance to look for surrounding text
            min_image_area: Minimum image area in pixels to process
            max_surrounding_text_chars: Maximum characters of surrounding text to include
            filter_header_footer: Whether to remove likely page headers/footers
            header_margin_ratio: Top page ratio treated as header area
            footer_margin_ratio: Bottom page ratio treated as footer area
            header_footer_min_repeat_pages: Minimum repeated pages before removing margin text
            max_header_footer_text_length: Max repeated margin text length to strip
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_text = extract_text
        self.use_vision_api = use_vision_api
        self.context_window_pixels = context_window_pixels
        self.min_image_area = min_image_area
        self.max_surrounding_text_chars = max_surrounding_text_chars
        self.filter_header_footer = filter_header_footer
        self.header_margin_ratio = max(0.0, min(header_margin_ratio, 0.3))
        self.footer_margin_ratio = max(0.0, min(footer_margin_ratio, 0.3))
        self.header_footer_min_repeat_pages = max(1, header_footer_min_repeat_pages)
        self.max_header_footer_text_length = max_header_footer_text_length
        self.vision_base_url = vision_base_url
        self.vision_model = vision_model or "gpt-4.1-mini"
        self.vision_provider = vision_provider.lower()
        self.vision_client = None
        
        if use_vision_api:
            self._init_vision_api(vision_api_key)
    
    def _init_vision_api(self, api_key: Optional[str]) -> None:
        """Initialize Vision API client"""
        try:
            key = (
                api_key
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("DEEPBRICKS_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
            
            if not key:
                logger.info("Vision API disabled: no API key configured.")
                self.vision_client = None
                return
            
            base_url = (
                self.vision_base_url
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("DEEPBRICKS_BASE_URL")
            )

            if self.vision_provider == "gemini":
                self.vision_model = (
                    self.vision_model
                    or os.getenv("VISION_MODEL")
                    or "gemini-2.5-flash"
                )

                try:
                    import google.genai as genai
                    from google.genai import types
                    self.vision_client = genai.Client(
                        api_key=api_key,   
                        http_options=types.HttpOptions(
                            base_url=base_url,   # ← 这里填你的 DeepBrick 或其他代理地址
                        )
                    )
                    logger.info(
                        f"Gemini Vision API initialized for image description using {self.vision_model}"
                    )
                except ImportError:
                    logger.error("google-generativeai package not installed. Install with: pip install google-generativeai")
                    self.vision_client = None
            else:
                # Initialize OpenAI-compatible client
                from openai import OpenAI

                if not key and base_url:
                    key = "dummy"

                client_kwargs = {"api_key": key}
                if base_url:
                    client_kwargs["base_url"] = base_url

                self.vision_client = OpenAI(**client_kwargs)
                self.vision_model = (
                    self.vision_model
                    or os.getenv("VISION_MODEL")
                    or os.getenv("LLM_MODEL")
                    or "gpt-4.1-mini"
                )
                logger.info(
                    f"OpenAI Vision API initialized for image description using {self.vision_model}"
                )
        except ImportError:
            logger.warning("OpenAI client not installed. Image descriptions will be basic.")
            self.vision_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Vision API: {e}")
            self.vision_client = None
    
    def process_pdf(
        self,
        file_path: str,
        doc_id: str,
        doc_title: str,
        language: str = "en"
    ) -> Document:
        """
        Process PDF with advanced layout understanding
        
        Args:
            file_path: Path to PDF file
            doc_id: Document ID
            doc_title: Document title
            language: Document language
        
        Returns:
            Document with extracted content blocks including images and context
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber required. Install with: pip install pdfplumber")
    
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        doc = Document(
            id=doc_id,
            title=doc_title,
            source_path=str(path),
            language=language
        )
        
        try:
            with pdfplumber.open(path) as pdf:
                # First pass: extract all layout elements
                all_text_blocks = []
                all_images = []
                page_heights: Dict[int, float] = {}
                
                table_blocks: List[TextBlock] = []
                for page_num, page in enumerate(pdf.pages):
                    page_heights[page_num] = float(getattr(page, "height", 0.0))
                    # Extract page-level text blocks first
                    page_text_blocks: List[TextBlock] = []
                    if self.extract_text:
                        page_text_blocks = self._extract_text_blocks(page, page_num)
                    
                    # Extract images with positions
                    if self.extract_images:
                        images = self._extract_images_from_page(
                            page, page_num, file_path, doc_id
                        )
                        all_images.extend(images)
                    
                    # Extract tables (filtered by image overlap)
                    table_bboxes: List[Tuple[float, float, float, float]] = []
                    if self.extract_tables:
                        # Get table locations using find_tables()
                        # Try multiple strategies to detect tables that might not have complete grid lines
                        table_objects = page.find_tables()
                        tables = page.extract_tables()
                        strategy_used = "default"
                        
                        # If no tables found with default strategy, try text-based detection
                        if not table_objects:
                            try:
                                # Use text-based detection for tables without clear grid lines
                                text_settings = {
                                    "horizontal_strategy": "text",
                                    "vertical_strategy": "text",
                                }
                                table_objects = page.find_tables(table_settings=text_settings)
                                tables = page.extract_tables(table_settings=text_settings)
                                strategy_used = "text"
                            except Exception as e:
                                logger.debug(f"Text-based table detection failed on page {page_num}: {e}")
                        
                        # Filter tables that overlap with images and format them
                        filtered_tables = self._filter_tables_from_image_overlap(
                            tables, table_objects, page_num, all_images, strategy_used
                        )
                        
                        # Store table blocks separately so they are not treated as normal text blocks
                        for table_text, table_bbox in filtered_tables:
                            table_blocks.append(TextBlock(
                                text=table_text,
                                page_num=page_num,
                                x0=table_bbox[0],
                                y0=table_bbox[1],
                                x1=table_bbox[2],
                                y1=table_bbox[3],
                                block_type="table"
                            ))
                            table_bboxes.append(table_bbox)

                    # Remove raw text blocks that overlap detected tables to avoid duplicate content
                    if page_text_blocks and table_bboxes:
                        page_text_blocks = self._filter_text_blocks_overlapping_tables(
                            page_text_blocks,
                            table_bboxes
                        )

                    all_text_blocks.extend(page_text_blocks)

                if self.filter_header_footer and all_text_blocks:
                    original_count = len(all_text_blocks)
                    all_text_blocks = self._filter_header_footer_blocks(
                        all_text_blocks,
                        page_heights=page_heights,
                    )
                    removed_count = original_count - len(all_text_blocks)
                    if removed_count:
                        logger.info(
                            f"Filtered {removed_count} header/footer text blocks from {file_path}"
                        )

                if self.filter_header_footer and all_images:
                    original_image_count = len(all_images)
                    all_images = self._filter_header_footer_images(
                        all_images,
                        page_heights=page_heights,
                    )
                    removed_image_count = original_image_count - len(all_images)
                    if removed_image_count:
                        logger.info(
                            f"Filtered {removed_image_count} header/footer images from {file_path}"
                        )
                    logger.debug(f"DEBUG: Images after filtering: {len(all_images)}/{original_image_count}")
                
                # Second pass: create content blocks with relationships
                # Process text blocks
                for text_block in all_text_blocks:
                    metadata = {
                        "block_type": text_block.block_type,
                        "position": {
                            "x0": text_block.x0,
                            "y0": text_block.y0,
                            "x1": text_block.x1,
                            "y1": text_block.y1
                        }
                    }
                    
                    # Add font size if available
                    if text_block.font_size is not None:
                        metadata["font_size"] = text_block.font_size
                    
                    block = ContentBlock(
                        id=f"{doc_id}_text_p{text_block.page_num}_{id(text_block)}",
                        type=ContentType.TEXT,
                        content=text_block.text,
                        modality=ModalityType.TEXT,
                        page_num=text_block.page_num,
                        language=language,
                        metadata=metadata
                    )
                    doc.add_content_block(block)

                # Process recognized tables separately so table text is not duplicated in normal text blocks
                for table_block in table_blocks:
                    metadata = {
                        "block_type": table_block.block_type,
                        "position": {
                            "x0": table_block.x0,
                            "y0": table_block.y0,
                            "x1": table_block.x1,
                            "y1": table_block.y1
                        }
                    }
                    block = ContentBlock(
                        id=f"{doc_id}_table_p{table_block.page_num}_{id(table_block)}",
                        type=ContentType.TEXT,
                        content=table_block.text,
                        modality=ModalityType.TEXT,
                        page_num=table_block.page_num,
                        language=language,
                        metadata=metadata
                    )
                    doc.add_content_block(block)

                # Process images with surrounding context
                for img_loc in all_images:
                    logger.debug(f"DEBUG: Processing image at page {img_loc.page_num}...")
                    try:
                        # Find surrounding text (returns dict with context_above and context_below)
                        context_info = self._get_surrounding_text(
                            img_loc, all_text_blocks
                        )
                        
                        # Generate image description
                        description = self._generate_image_description(
                            img_loc, context_info
                        )
                        
                        # Create image content block
                        block = ContentBlock(
                            id=f"{doc_id}_image_p{img_loc.page_num}_{id(img_loc)}",
                            type=ContentType.IMAGE,
                            content=img_loc.image_path,
                            modality=ModalityType.VISUAL,
                            page_num=img_loc.page_num,
                            language=language,
                            metadata={
                                "description": description,
                                "context_above": context_info.get("context_above", ""),
                                "context_below": context_info.get("context_below", ""),
                                "position": {
                                    "x0": img_loc.x0,
                                    "y0": img_loc.y0,
                                    "x1": img_loc.x1,
                                    "y1": img_loc.y1,
                                    "width": img_loc.width,
                                    "height": img_loc.height
                                },
                                "related_text_block_count": len(
                                    [t for t in all_text_blocks 
                                     if abs(t.page_num - img_loc.page_num) <= 1]
                                )
                            }
                        )
                        doc.add_content_block(block)
                    except Exception as e:
                        logger.error(f"ERROR processing image at page {img_loc.page_num}: {e}")
                        raise
            
            logger.info(
                f"Successfully processed PDF: {file_path} "
                f"({len(doc.content_blocks)} content blocks)"
            )
            return doc
        
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    def _extract_text_blocks(
        self,
        page: Any,
        page_num: int
    ) -> List[TextBlock]:
        """Extract text blocks from page with font size information"""
        blocks = []
        try:
            # Use extract_words for better font size extraction
            if hasattr(page, "extract_words"):
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
                
                # Group words into text blocks by similar properties
                text_groups = self._group_words_into_blocks(words, page_num)
                blocks.extend(text_groups)
            
            # Fallback to extract_text_lines if extract_words not available
            elif hasattr(page, "extract_text_lines"):
                for line in page.extract_text_lines(strip=True) or []:
                    text = str(line.get("text", "")).strip()
                    if not text:
                        continue
                    blocks.append(
                        TextBlock(
                            text=text,
                            page_num=page_num,
                            x0=float(line.get("x0", 0.0)),
                            y0=float(line.get("top", line.get("y0", 0.0))),
                            x1=float(line.get("x1", getattr(page, "width", 0.0))),
                            y1=float(line.get("bottom", line.get("y1", getattr(page, "height", 0.0)))),
                            block_type="text"
                        )
                    )
            
            if not blocks:
                text = page.extract_text()
                if text:
                    blocks.append(
                        TextBlock(
                            text=text,
                            page_num=page_num,
                            x0=0,
                            y0=0,
                            x1=page.width,
                            y1=page.height,
                            block_type="text"
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to extract layout/text blocks: {e}. Using fallback.")
            text = page.extract_text()
            if text:
                blocks.append(
                    TextBlock(
                        text=text,
                        page_num=page_num,
                        x0=0,
                        y0=0,
                        x1=page.width,
                        y1=page.height,
                        block_type="text"
                    )
                )
        
        return blocks
    
    def _group_words_into_blocks(
        self,
        words: List[Dict[str, Any]],
        page_num: int
    ) -> List[TextBlock]:
        """Group words into text blocks based on position and font properties"""
        if not words:
            return []
        
        blocks = []
        
        # Sort words by position (top to bottom, left to right)
        sorted_words = sorted(words, key=lambda w: (w.get('top', 0), w.get('x0', 0)))
        
        current_block_words = []
        current_font_size = None
        current_y = None
        line_tolerance = 5  # pixels tolerance for line grouping
        
        for word in sorted_words:
            word_text = word.get('text', '').strip()
            if not word_text:
                continue
                
            word_y = word.get('top', 0)
            word_font_size = word.get('height', None)  # pdfplumber uses 'height' for font size
            
            # Start a new block when we hit a new line or a clear font-size change.
            is_new_line = current_y is not None and abs(word_y - current_y) > line_tolerance
            font_size_changed = (
                current_font_size is not None and 
                word_font_size is not None and
                abs(word_font_size - current_font_size) > 2
            )
            
            if current_block_words and (is_new_line or font_size_changed):
                block_text = ' '.join(w['text'] for w in current_block_words)
                block_x0 = min(w['x0'] for w in current_block_words)
                block_y0 = min(w['top'] for w in current_block_words)
                block_x1 = max(w['x1'] for w in current_block_words)
                block_y1 = max(w['bottom'] for w in current_block_words)
                
                blocks.append(TextBlock(
                    text=block_text,
                    page_num=page_num,
                    x0=block_x0,
                    y0=block_y0,
                    x1=block_x1,
                    y1=block_y1,
                    block_type="text",
                    font_size=current_font_size
                ))
                current_block_words = []
                current_font_size = word_font_size
                current_y = word_y
            elif current_font_size is None:
                current_font_size = word_font_size
                current_y = word_y
            
            current_block_words.append(word)
        
        # Add final block
        if current_block_words:
            block_text = ' '.join(w['text'] for w in current_block_words)
            block_x0 = min(w['x0'] for w in current_block_words)
            block_y0 = min(w['top'] for w in current_block_words)
            block_x1 = max(w['x1'] for w in current_block_words)
            block_y1 = max(w['bottom'] for w in current_block_words)
            
            blocks.append(TextBlock(
                text=block_text,
                page_num=page_num,
                x0=block_x0,
                y0=block_y0,
                x1=block_x1,
                y1=block_y1,
                block_type="text",
                font_size=current_font_size
            ))
        
        return blocks
    
    def _filter_tables_from_image_overlap(
        self,
        tables: List[List[List[str]]],
        table_objects: List[Any],
        page_num: int,
        all_images: List[ImageLocation],
        strategy_used: str = "default",
    ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """
        Filter out tables that are completely or mostly inside images.
        
        Tables that are located within image boundaries are excluded since they're 
        already visually represented by the image (e.g., tables in screenshots).
        Also filters out list-like structures that aren't real tables.
        
        Args:
            tables: List of extracted tables (from page.extract_tables())
            table_objects: Corresponding table objects with position info (from page.find_tables())
            page_num: Current page number
            all_images: All extracted images from the document
            strategy_used: "default" or "text" - which detection strategy was used
        
        Returns:
            List of tuples (formatted_table_text, bbox) for tables not completely inside images
        """
        filtered_tables = []
        
        if not tables or not table_objects:
            return filtered_tables
        
        # Process each table with its bounding box
        for table, table_obj in zip(tables, table_objects):
            table_bbox = table_obj.bbox  # (x0, y0, x1, y1)
            
            # Filter out list-like structures (single column with bullet points/numbered items)
            # This applies to both default and text-based detection results
            if not self._is_valid_table(table):
                logger.debug(f"Skipping list-like structure at page {page_num} bbox={table_bbox}")
                continue
            
            # Check if this table is completely inside any image on the same page
            is_inside_image = False
            
            for img in all_images:
                if img.page_num == page_num:
                    # Check if table is completely inside image bounds
                    if self._is_table_completely_inside_image(table_bbox, img):
                        is_inside_image = True
                        logger.debug(
                            f"Table at page {page_num} bbox={table_bbox} "
                            f"is completely inside image bbox=({img.x0}, {img.y0}, {img.x1}, {img.y1})"
                        )
                        break
            
            # Include table only if it's not inside any image
            if not is_inside_image:
                table_text = self._format_table(table)
                filtered_tables.append((table_text, table_bbox))
        
        return filtered_tables
    
    def _filter_text_blocks_overlapping_tables(
        self,
        text_blocks: List[TextBlock],
        table_bboxes: List[Tuple[float, float, float, float]],
        overlap_threshold: float = 0.8
    ) -> List[TextBlock]:
        """
        Remove text blocks that strongly overlap detected table regions.

        This prevents duplicate text content when a table has already been
        extracted and represented as a separate table block.
        """
        if not text_blocks or not table_bboxes:
            return text_blocks

        filtered = []
        for block in text_blocks:
            if block.block_type != "text":
                filtered.append(block)
                continue

            block_bbox = (block.x0, block.y0, block.x1, block.y1)
            if any(
                self._calculate_bbox_overlap_ratio(block_bbox, table_bbox) >= overlap_threshold
                for table_bbox in table_bboxes
            ):
                logger.debug(
                    f"Skipping text block overlapping detected table on page {block.page_num}: "
                    f"bbox={block_bbox}"
                )
                continue

            filtered.append(block)

        return filtered

    def _is_valid_table(self, table: List[List[str]]) -> bool:
        """
        Determine if extracted content is a valid table or just a list.
        
        Used for text-based detection strategy only. A valid table should have:
        - At least 2 columns (single column with 2+ rows is likely a list/bullets)
        - OR at least 2 rows and meaningful content spread across multiple columns
        
        Args:
            table: Table content as list of rows, each row is list of cells
        
        Returns:
            True if it's a valid table, False if it's a list-like structure
        """
        if not table or len(table) < 1:
            return False
        
        # Single row is not a table
        if len(table) == 1:
            return False
        
        # Get number of columns from headers and other rows
        col_counts = [len(row) for row in table]
        num_cols = max(col_counts) if col_counts else 0
        
        # A single-column structure is likely a list, not a table
        if num_cols <= 1:
            return False
        
        # Check if columns have meaningful content (not mostly empty)
        non_empty_cols = set()
        for row in table:
            for col_idx, cell in enumerate(row):
                if cell and cell.strip():
                    non_empty_cols.add(col_idx)
        
        # If at least 2 columns have content, likely a real table
        if len(non_empty_cols) >= 2:
            return True
        
        # If only 1 column has content in a multi-column structure,
        # it's probably a list that was mistakenly detected as a table
        if len(non_empty_cols) <= 1 and num_cols > 2:
            return False
        
        return True
    
    def _is_table_completely_inside_image(
        self,
        table_bbox: Tuple[float, float, float, float],
        image_loc: ImageLocation
    ) -> bool:
        """
        Check if a table is substantially inside an image.
        
        A table is considered inside an image if a significant portion of its area
        is within the image bounds. This handles cases where:
        - Table entirely within image (100% inside)
        - Table mostly within image (>90% inside) - represents tables shown as images
        
        We use 90% threshold to avoid filtering legitimate tables that happen to
        be partially overlapped by images. A 90% overlap strongly indicates the
        table is actually part of an image artifact, not a separate table.
        
        Args:
            table_bbox: Table bounding box as (x0, y0, x1, y1)
            image_loc: Image location object
        
        Returns:
            True if table is substantially (>90%) inside the image
        """
        tx0, ty0, tx1, ty1 = table_bbox
        ix0, iy0, ix1, iy1 = image_loc.x0, image_loc.y0, image_loc.x1, image_loc.y1
        
        # Calculate intersection
        inter_x0 = max(tx0, ix0)
        inter_y0 = max(ty0, iy0)
        inter_x1 = min(tx1, ix1)
        inter_y1 = min(ty1, iy1)
        
        # No intersection
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return False
        
        # Calculate areas
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        table_area = (tx1 - tx0) * (ty1 - ty0)
        
        if table_area < 1:
            return False
        
        # Return True if table is >90% inside image
        inside_ratio = inter_area / table_area
        return inside_ratio > 0.90

    def _filter_header_footer_blocks(
        self,
        text_blocks: List[TextBlock],
        page_heights: Dict[int, float],
    ) -> List[TextBlock]:
        """
        Remove likely header/footer text using advanced position and overlap analysis.
        
        Strategy:
        1. Identify candidates in header/footer zones
        2. Cluster candidates by position and content similarity
        3. Filter clusters with high spatial consistency across pages
        4. Preserve content that doesn't match header/footer pattern
        """
        if not self.filter_header_footer or not text_blocks:
            return text_blocks

        # Step 1: Identify margin candidates
        margin_candidates: List[Tuple[int, TextBlock]] = []
        for idx, block in enumerate(text_blocks):
            if block.block_type != "text":
                continue
            page_height = page_heights.get(block.page_num, 0.0)
            if not self._is_header_footer_zone(block, page_height):
                continue
            margin_candidates.append((idx, block))

        if not margin_candidates:
            return text_blocks

        # Step 2: Cluster candidates by position and content
        # Separate into headers and footers
        header_clusters = self._cluster_margin_elements(
            [b for _, b in margin_candidates],
            page_heights,
            is_header=True
        )
        footer_clusters = self._cluster_margin_elements(
            [b for _, b in margin_candidates],
            page_heights,
            is_header=False
        )

        # Step 3: Identify elements to remove
        blocks_to_remove = set()
        
        # Process header clusters
        for cluster in header_clusters:
            if self._is_repeated_margin_element(cluster, page_heights):
                for block in cluster:
                    # Find index in original list
                    for idx, b in margin_candidates:
                        if b is block:
                            blocks_to_remove.add(idx)
                            break

        # Process footer clusters
        for cluster in footer_clusters:
            if self._is_repeated_margin_element(cluster, page_heights):
                for block in cluster:
                    for idx, b in margin_candidates:
                        if b is block:
                            blocks_to_remove.add(idx)
                            break

        # Step 4: Return filtered blocks
        filtered_blocks = [
            block for idx, block in enumerate(text_blocks)
            if idx not in blocks_to_remove
        ]
        return filtered_blocks

    def _filter_header_footer_images(
        self,
        images: List[ImageLocation],
        page_heights: Dict[int, float],
    ) -> List[ImageLocation]:
        """
        Remove repeated logo-like images from page headers/footers using IoU clustering.
        
        Uses position-based analysis to identify images that appear in consistent
        locations across multiple pages (typical of logos, page numbers, etc).
        """
        if not self.filter_header_footer or not images:
            return images

        # Cluster images by position in margin zones
        header_clusters = self._cluster_margin_images(
            images, page_heights, is_header=True
        )
        footer_clusters = self._cluster_margin_images(
            images, page_heights, is_header=False
        )

        images_to_remove = set()

        # Check each cluster for repeated margin elements
        for cluster_idx, cluster in enumerate(header_clusters + footer_clusters):
            if self._is_repeated_margin_image_cluster(cluster, page_heights):
                for img in cluster:
                    # Find index in original list
                    for list_idx, original_img in enumerate(images):
                        if original_img is img:
                            images_to_remove.add(list_idx)
                            break

        filtered_images = [
            img for idx, img in enumerate(images)
            if idx not in images_to_remove
        ]
        return filtered_images

    def _cluster_margin_images(
        self,
        images: List[ImageLocation],
        page_heights: Dict[int, float],
        is_header: bool = True
    ) -> List[List[ImageLocation]]:
        """
        Cluster margin images by position similarity.
        
        Images with high spatial overlap across pages are grouped,
        indicating they're the same header/footer element.
        """
        if not images:
            return []

        # Filter to only header or footer images
        filtered = []
        for img in images:
            page_height = page_heights.get(img.page_num, 0.0)
            in_header = float(img.y0) <= page_height * self.header_margin_ratio
            in_footer = float(img.y1) >= page_height * (1 - self.footer_margin_ratio)

            if is_header and in_header:
                filtered.append(img)
            elif not is_header and in_footer:
                filtered.append(img)

        if not filtered:
            return []

        # Cluster by position (IoU > threshold)
        clusters: List[List[ImageLocation]] = []
        used = set()

        for i, img_i in enumerate(filtered):
            if i in used:
                continue

            cluster = [img_i]
            used.add(i)

            for j, img_j in enumerate(filtered[i + 1:], start=i + 1):
                if j in used:
                    continue

                iou = self._calculate_image_position_iou(img_i, img_j)
                if iou > 0.5:
                    cluster.append(img_j)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _calculate_image_position_iou(
        self,
        img1: ImageLocation,
        img2: ImageLocation
    ) -> float:
        """
        Calculate position-based IoU for two images.
        
        Measures spatial overlap: high IoU indicates images appear in
        the same location (likely duplicate header/footer elements).
        """
        # Get bounding boxes
        x0_1, y0_1, x1_1, y1_1 = img1.get_bbox()
        x0_2, y0_2, x1_2, y1_2 = img2.get_bbox()

        # Calculate intersection
        inter_x0 = max(x0_1, x0_2)
        inter_y0 = max(y0_1, y0_2)
        inter_x1 = min(x1_1, x1_2)
        inter_y1 = min(y1_1, y1_2)

        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return 0.0

        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

        # Calculate union
        area1 = img1.get_area()
        area2 = img2.get_area()
        union_area = area1 + area2 - inter_area

        if union_area < 1:
            return 0.0

        return inter_area / union_area

    def _is_repeated_margin_image_cluster(
        self,
        cluster: List[ImageLocation],
        page_heights: Dict[int, float]
    ) -> bool:
        """
        Determine if an image cluster represents repeated margin elements (logo, footer, etc).
        
        Uses consistency analysis:
        1. Appears on multiple pages
        2. Consistent position across pages
        3. Similar size across pages
        4. Small enough to be a margin decoration (not large content)
        """
        if len(cluster) < self.header_footer_min_repeat_pages:
            return False

        unique_pages = len(set(img.page_num for img in cluster))
        if unique_pages < self.header_footer_min_repeat_pages:
            return False

        # IMPORTANT: Exclude large content images
        # If any image in cluster is > 5% of page area, don't treat as margin element
        avg_page_area = page_heights.get(cluster[0].page_num, 800.0) * 595.0  # ~A4 standard
        for img in cluster:
            img_area = img.get_area()
            area_ratio = img_area / avg_page_area if avg_page_area > 0 else 0
            if area_ratio > 0.05:  # > 5% of page = content, not margin
                logger.debug(f"Skipping margin filter for large image cluster (area_ratio={area_ratio:.1%})")
                return False

        # Calculate position consistency
        consistency = self._image_position_consistency_score(cluster)

        # Remove if highly consistent position (typical of repeated margin images)
        if consistency > 0.75 and unique_pages >= self.header_footer_min_repeat_pages:
            logger.debug(f"Removing repeated margin image cluster (consistency={consistency:.3f})")
            return True

        return False

    def _image_position_consistency_score(
        self,
        cluster: List[ImageLocation]
    ) -> float:
        """
        Calculate position consistency for image cluster.
        
        Measures how much position and size vary across cluster members.
        High consistency (close to 1.0) indicates the same element repeated.
        """
        if len(cluster) < 2:
            return 0.0

        # Collect positions and sizes
        x0_vals = [float(img.x0) for img in cluster]
        y0_vals = [float(img.y0) for img in cluster]
        widths = [float(img.width) for img in cluster]
        heights = [float(img.height) for img in cluster]

        # Calculate standard deviations (normalized)
        def calc_consistency(values, normalizer=1.0):
            if not values or normalizer < 1:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            return max(0.0, 1.0 - (std / max(normalizer, 1.0)))

        # Normalize by typical document dimensions
        x_consistency = calc_consistency(x0_vals, 600.0)
        y_consistency = calc_consistency(y0_vals, 800.0)
        width_consistency = calc_consistency(widths, max(widths) if widths else 1.0)
        height_consistency = calc_consistency(heights, max(heights) if heights else 1.0)

        # Weight: position more important than size
        consistency = (
            0.4 * x_consistency +
            0.4 * y_consistency +
            0.1 * width_consistency +
            0.1 * height_consistency
        )

        return consistency

    def _is_header_footer_zone(self, block: Any, page_height: float) -> bool:
        """Check whether a text/image block is inside the top/bottom margin area."""
        if page_height <= 0:
            return False

        header_limit = page_height * self.header_margin_ratio
        footer_limit = page_height * (1 - self.footer_margin_ratio)
        return float(block.y0) <= header_limit or float(block.y1) >= footer_limit

    def _cluster_margin_elements(
        self,
        blocks: List[TextBlock],
        page_heights: Dict[int, float],
        is_header: bool = True
    ) -> List[List[TextBlock]]:
        """
        Cluster margin elements by position similarity using IoU clustering.
        
        Elements with high spatial overlap across pages are grouped together,
        indicating they are likely the same header/footer appearing on multiple pages.
        
        Args:
            blocks: Text blocks in margin zones
            page_heights: Height of each page
            is_header: True to cluster headers (top margin), False for footers (bottom)
        
        Returns:
            List of clusters, where each cluster is a list of similar margin blocks
        """
        if not blocks:
            return []

        # Filter to only header or footer blocks
        filtered = []
        for block in blocks:
            page_height = page_heights.get(block.page_num, 0.0)
            in_header = float(block.y0) <= page_height * self.header_margin_ratio
            in_footer = float(block.y1) >= page_height * (1 - self.footer_margin_ratio)
            
            if is_header and in_header:
                filtered.append(block)
            elif not is_header and in_footer:
                filtered.append(block)

        if not filtered:
            return []

        # Cluster by position similarity (IoU > threshold)
        clusters: List[List[TextBlock]] = []
        used = set()

        for i, block_i in enumerate(filtered):
            if i in used:
                continue

            cluster = [block_i]
            used.add(i)

            for j, block_j in enumerate(filtered[i + 1:], start=i + 1):
                if j in used:
                    continue

                # Calculate spatial similarity (position-based IoU)
                iou = self._calculate_position_iou(block_i, block_j)
                if iou > 0.5:  # High position similarity threshold
                    cluster.append(block_j)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _calculate_position_iou(self, block1: TextBlock, block2: TextBlock) -> float:
        """
        Calculate IoU (Intersection over Union) of two block positions.
        
        This measures how much two blocks overlap in space relative to their combined area.
        High IoU means they are likely the same element appearing on different pages.
        
        Returns:
            Intersection over Union ratio (0.0 to 1.0)
        """
        # Normalize coordinates to page width (assume standard page width)
        x0_1, x1_1 = float(block1.x0), float(block1.x1)
        x0_2, x1_2 = float(block2.x0), float(block2.x1)
        
        # Width should be similar for margin elements 
        width1 = x1_1 - x0_1
        width2 = x1_2 - x0_2

        if width1 < 1 or width2 < 1:
            return 0.0

        # Intersection width
        inter_x0 = max(x0_1, x0_2)
        inter_x1 = min(x1_1, x1_2)
        inter_width = max(0, inter_x1 - inter_x0)

        # Union width
        union_x0 = min(x0_1, x0_2)
        union_x1 = max(x1_1, x1_2)
        union_width = union_x1 - union_x0

        if union_width < 1:
            return 0.0

        return inter_width / union_width

    def _is_repeated_margin_element(
        self,
        cluster: List[TextBlock],
        page_heights: Dict[int, float]
    ) -> bool:
        """
        Determine if a cluster represents a truly repeated margin element.
        
        Uses multiple heuristics:
        1. Appears on multiple pages (frequency check)
        2. Consistent position across pages (position consistency score)
        3. Similar text content or purely numeric (pattern matching)
        4. Text length within typical margin range
        
        Args:
            cluster: List of blocks in this cluster
            page_heights: Page height information
        
        Returns:
            True if this cluster should be removed as header/footer
        """
        if len(cluster) < self.header_footer_min_repeat_pages:
            return False

        # Heuristic 1: Frequency - appears in enough pages
        unique_pages = len(set(b.page_num for b in cluster))
        if unique_pages < self.header_footer_min_repeat_pages:
            return False

        # Heuristic 2: Position consistency
        pos_consistency = self._position_consistency_score(cluster)
        
        # Heuristic 3: Content analysis
        has_page_marker = any(self._looks_like_page_marker(b.text) for b in cluster)
        has_long_repeated_text = self._has_long_repeated_content(cluster)

        # Heuristic 4: Text length check  
        avg_text_len = sum(len(b.text.strip()) for b in cluster) / len(cluster) if cluster else 0

        # Determine whether this repeated cluster sits very close to page margins.
        page_height = page_heights.get(cluster[0].page_num, 0.0) if cluster else 0.0
        tight_header_limit = page_height * self.header_margin_ratio * 0.3
        tight_footer_limit = page_height * (1 - self.footer_margin_ratio * 0.3)
        avg_y0 = sum(float(b.y0) for b in cluster) / len(cluster) if cluster else 0.0
        avg_y1 = sum(float(b.y1) for b in cluster) / len(cluster) if cluster else 0.0
        in_tight_header_zone = avg_y0 <= tight_header_limit if page_height > 0 else False
        in_tight_footer_zone = avg_y1 >= tight_footer_limit if page_height > 0 else False

        # Decision logic: remove if it meets these criteria
        should_remove = (
            (pos_consistency > 0.7 and has_page_marker)  # Clear page marker pattern
            or (
                (has_long_repeated_text or (pos_consistency > 0.8 and avg_text_len <= self.max_header_footer_text_length))
                and (has_page_marker or in_tight_header_zone or in_tight_footer_zone)
            )
        )

        return should_remove

    def _position_consistency_score(self, cluster: List[TextBlock]) -> float:
        """
        Calculate how consistently positioned the blocks in a cluster are.
        
        If blocks appear in nearly the same position across pages, they likely
        represent the same header/footer element repeated on multiple pages.
        
        Returns:
            Consistency score from 0.0 (very inconsistent) to 1.0 (perfectly consistent)
        """
        if len(cluster) < 2:
            return 0.0

        # Calculate position variance
        x_positions = [float(b.x0) for b in cluster]
        widths = [float(b.x1 - b.x0) for b in cluster]

        # Standard deviation of x-position (normalized by typical page width)
        x_mean = sum(x_positions) / len(x_positions)
        x_variance = sum((x - x_mean) ** 2 for x in x_positions) / len(x_positions)
        x_std = math.sqrt(x_variance)

        # Normalize by page width (assume ~600 pixel typical width at extraction resolution)
        x_consistency = max(0.0, 1.0 - (x_std / 600.0))

        # Width consistency
        width_mean = sum(widths) / len(widths)
        width_variance = sum((w - width_mean) ** 2 for w in widths) / len(widths)
        width_std = math.sqrt(width_variance)
        width_consistency = max(0.0, 1.0 - (width_std / width_mean)) if width_mean > 0 else 0.0

        # Combine: position consistency is more important than width consistency
        consistency = 0.7 * x_consistency + 0.3 * width_consistency

        logger.debug(
            f"Cluster position consistency: {consistency:.3f} "
            f"(x_std={x_std:.1f}, width_std={width_std:.1f})"
        )
        return consistency

    def _has_long_repeated_content(self, cluster: List[TextBlock]) -> bool:
        """
        Check if cluster has repeated text content (beyond just page numbers).
        
        Repeated content appearing in multiple pages is a strong indicator of
        header/footer elements, especially when normalized forms are identical.
        """
        if len(cluster) < 2:
            return False

        # Normalize all text
        normalized_texts = [self._normalize_margin_text(b.text) for b in cluster]
        normalized_texts = [t for t in normalized_texts if t]  # Remove empty

        if not normalized_texts:
            return False

        # Check if most blocks have the same normalized text
        from collections import Counter
        text_counts = Counter(normalized_texts)
        most_common_text, count = text_counts.most_common(1)[0]

        # If same normalized text appears in at least header_footer_min_repeat_pages
        repeated = count >= self.header_footer_min_repeat_pages and most_common_text

        return repeated

    def _normalize_margin_text(self, text: str) -> str:
        """
        Normalize candidate header/footer text for repeat detection.
        
        This enables comparison of text that may vary slightly across pages
        (e.g., different page numbers, but same template text).
        """
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        # Replace numbers with # to match patterns (e.g., "page 1" and "page 2" both become "page #")
        normalized = re.sub(r"\d+", "#", normalized)
        # Remove special characters
        normalized = re.sub(r"[^\w\s#\u4e00-\u9fff]", " ", normalized)
        # Clean up whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:240]

    def _looks_like_page_marker(self, text: str) -> bool:
        """
        Detect if text looks like a page number or page marker.
        
        Examples:
        - "Page 1", "P. 1", "1/10", "第1页"
        - Standalone numbers or dashes
        """
        compact = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not compact:
            return True

        # Very short and mostly numbers/dashes
        if len(compact) <= 12 and re.fullmatch(r"[-–—\s\d/]+", compact):
            return True

        # Page number patterns (English and Chinese)
        page_patterns = (
            r"^(page|p\.?)[\s#:.-]*[\divxlcdm]+(?:\s*(?:/|of)\s*[\divxlcdm]+)?$",
            r"^第\s*[\divxlcdm]+\s*页$",
            r"^[\divxlcdm]+\s*/\s*[\divxlcdm]+$",
        )
        return any(re.fullmatch(pattern, compact) for pattern in page_patterns)
    
    def _extract_images_from_page(
        self,
        page: Any,
        page_num: int,
        pdf_path: str,
        doc_id: str
    ) -> List[ImageLocation]:
        """Extract images from PDF page"""
        images = []
        try:
            # Get image objects from page
            image_objects = page.images
            
            for img_idx, img in enumerate(image_objects):
                try:
                    # Check image size
                    img_area = (img["x1"] - img["x0"]) * (img["y1"] - img["y0"])
                    if img_area < self.min_image_area:
                        logger.debug(f"Skipping small image: {img_area} pixels")
                        continue
                    
                    # Save image to file
                    image_path = self._extract_and_save_image(
                        page, img, page_num, pdf_path, doc_id, img_idx
                    )
                    
                    if image_path:
                        # Use top/bottom if available, otherwise convert bottom-origin y0/y1 to top-origin coordinates.
                        left = float(img.get("x0", 0.0))
                        right = float(img.get("x1", left))
                        page_height = float(getattr(page, "height", 0.0))
                        if "top" in img and "bottom" in img:
                            top = float(img.get("top", 0.0))
                            bottom = float(img.get("bottom", 0.0))
                        else:
                            raw_y0 = float(img.get("y0", 0.0))
                            raw_y1 = float(img.get("y1", 0.0))
                            if page_height > 0:
                                top = page_height - raw_y1
                                bottom = page_height - raw_y0
                            else:
                                top = raw_y0
                                bottom = raw_y1
                        if top > bottom:
                            top, bottom = bottom, top
                        img_loc = ImageLocation(
                            image_path=image_path,
                            page_num=page_num,
                            x0=left,
                            y0=top,
                            x1=right,
                            y1=bottom,
                            width=right - left,
                            height=bottom - top
                        )
                        images.append(img_loc)
                        logger.debug(f"Extracted image: {image_path}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_num}: {e}")
        
        return images
    
    def _extract_and_save_image(
        self,
        page: Any,
        img_obj: Dict[str, Any],
        page_num: int,
        pdf_path: str,
        doc_id: str,
        img_idx: int
    ) -> Optional[str]:
        """Extract image from page and save to disk"""
        try:
            from PIL import Image
            
            pdf_path = Path(pdf_path)
            images_dir = pdf_path.parent / f"{doc_id}_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            image_path = images_dir / f"page{page_num}_img{img_idx}.png"
            
            # Preferred path: decode the original PDF image stream directly.
            stream = img_obj.get("stream")
            if stream is not None and hasattr(stream, "get_data"):
                try:
                    raw_bytes = stream.get_data()
                    image = Image.open(io.BytesIO(raw_bytes))
                    if image.mode not in ("RGB", "RGBA", "L"):
                        image = image.convert("RGB")
                    image.save(image_path, "PNG")
                    return str(image_path)
                except Exception as stream_error:
                    logger.debug(
                        f"Direct image stream extraction failed for page {page_num}, "
                        f"image {img_idx}: {stream_error}"
                    )
            
            # Fallback: render the image region from the page.
            bbox = (
                float(img_obj.get("x0", 0.0)),
                float(img_obj.get("top", img_obj.get("y0", 0.0))),
                float(img_obj.get("x1", getattr(page, "width", 0.0))),
                float(img_obj.get("bottom", img_obj.get("y1", getattr(page, "height", 0.0)))),
            )
            rendered = page.crop(bbox).to_image(resolution=150)
            
            if hasattr(rendered, "original") and hasattr(rendered.original, "save"):
                rendered.original.save(image_path, "PNG")
            else:
                rendered.save(str(image_path))
            
            return str(image_path)
        except ImportError:
            logger.warning("Pillow is required for PDF image extraction. Install with: pip install Pillow")
            return None
        except Exception as e:
            logger.warning(f"Failed to save image: {e}")
            return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences by common punctuation"""
        if not text:
            return []
        
        # Split by sentence terminators: . ! ? ; 。 ！ ？ ； \n
        # Keep delimiters with sentences for proper context
        pattern = r"([.。！？!?；;:\n])"
        parts = re.split(pattern, text)
        
        sentences = []
        buf = ""
        for part in parts:
            if not part:
                continue
            if re.fullmatch(pattern, part):
                buf += part
                if buf.strip():
                    sentences.append(buf)
                buf = ""
            else:
                buf += part
        
        if buf.strip():
            sentences.append(buf)
        
        return sentences
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text (approximation)"""
        if not text:
            return 0
        # Simple approximation: ~4 chars per token on average
        # For more accuracy, could use tiktoken or other token counter
        return max(1, len(text) // 4)
    
    def _get_surrounding_text(
        self,
        image_loc: ImageLocation,
        text_blocks: List[TextBlock]
    ) -> Dict[str, str]:
        """
        Get text surrounding an image using position-based selection.
        
        Returns separate context_above and context_below for better context management.
        
        Optimized to avoid unnecessary context by:
        1. Selecting text blocks closest to image position
        2. Splitting into sentences and respecting token budget
        3. Prioritizing same-page content over adjacent pages
        4. Keeping above and below context separate
        
        Args:
            image_loc: Image location
            text_blocks: All text blocks in document
        
        Returns:
            Dict with "context_above" and "context_below" keys
        """
        result = {"context_above": "", "context_below": ""}
        
        if not text_blocks or self.max_surrounding_text_chars <= 0:
            return result
        
        # Separate text blocks by page and position
        same_page_blocks = []
        adjacent_page_blocks = []
        
        for text_block in text_blocks:
            if text_block.page_num == image_loc.page_num:
                same_page_blocks.append(text_block)
            elif abs(text_block.page_num - image_loc.page_num) == 1:
                adjacent_page_blocks.append(text_block)
        
        # Get image center position for distance calculation
        img_center_y = (image_loc.y0 + image_loc.y1) / 2.0
        img_center_x = (image_loc.x0 + image_loc.x1) / 2.0
        
        # Find closest text blocks (upper and lower context)
        upper_blocks = []
        lower_blocks = []
        
        # Prioritize same-page blocks and ignore page markers
        for block in same_page_blocks:
            block_text = block.text.strip()
            if not block_text or self._looks_like_page_marker(block_text):
                continue
            block_center_y = (block.y0 + block.y1) / 2.0
            if block_center_y < img_center_y:  # Above image
                upper_blocks.append(block)
            elif block_center_y > img_center_y:  # Below image
                lower_blocks.append(block)
        
        # Sort by distance to image center
        upper_blocks.sort(key=lambda b: (img_center_y - (b.y0 + b.y1) / 2.0))
        lower_blocks.sort(key=lambda b: ((b.y0 + b.y1) / 2.0 - img_center_y))
        
        # Build context with token budget (split between above and below)
        token_budget = max(10, self.max_surrounding_text_chars // 4)
        half_budget = token_budget // 2
        
        # Collect above context (from far to near)
        upper_text = self._collect_context_sentences(upper_blocks, half_budget, from_end=True)
        # Collect below context (from near to far)
        lower_text = self._collect_context_sentences(lower_blocks, half_budget, from_end=False)
        
        # Truncate each context individually if needed
        max_per_context = self.max_surrounding_text_chars // 2
        
        if upper_text and len(upper_text) > max_per_context:
            upper_text = upper_text[:max_per_context].rsplit(" ", 1)[0] + "..."
        
        if lower_text and len(lower_text) > max_per_context:
            lower_text = lower_text[:max_per_context].rsplit(" ", 1)[0] + "..."
        
        result["context_above"] = upper_text.strip()
        result["context_below"] = lower_text.strip()
        
        return result
    
    def _collect_context_sentences(
        self,
        text_blocks: List[TextBlock],
        token_budget: int,
        from_end: bool = False
    ) -> str:
        """
        Collect sentences from text blocks respecting token budget.
        
        Blocks are ordered by distance (closest first).
        For upper context (from_end=True): collect far-to-near (reverse blocks)
        For lower context (from_end=False): collect near-to-far (keep order)
        
        Args:
            text_blocks: List of text blocks ordered by distance (closest first)
            token_budget: Maximum tokens to collect
            from_end: If True, reverse block order (upper context, far-to-near)
                     If False, keep normal order (lower context, near-to-far)
        
        Returns:
            Collected context text in correct sentence order
        """
        collected = []
        remaining_tokens = token_budget
        
        # For upper context: reverse block order to go from far-to-near
        # For lower context: keep block order to go from near-to-far
        blocks_to_process = reversed(text_blocks) if from_end else text_blocks
        
        for block in blocks_to_process:
            if remaining_tokens <= 0:
                break
            
            text = block.text.strip()
            if not text:
                continue
            
            # Split into sentences (will be in correct order from splitting)
            sentences = self._split_sentences(text)
            if not sentences:
                continue
            
            # Collect sentences in normal order - block order handles the direction
            for sentence in sentences:
                if remaining_tokens <= 0:
                    break
                
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                token_count = self._count_tokens(sentence)
                if token_count <= remaining_tokens:
                    collected.append(sentence)
                    remaining_tokens -= token_count
                else:
                    # Only take partial sentence if needed
                    collected.append(sentence[:remaining_tokens * 4])
                    remaining_tokens = 0
                    break
        
        # Simple join - block order and sentence order are already correct
        return " ".join(collected).strip()
    
    def _generate_image_description(
        self,
        image_loc: ImageLocation,
        context_info: Dict[str, str]
    ) -> str:
        """
        Generate description for image
        
        Args:
            image_loc: Image location
            context_info: Dict with "context_above" and "context_below" keys
        
        Returns:
            Image description
        """
        # Try to use Vision API if available
        if self.vision_client:
            try:
                description = self._describe_image_with_vision(
                    image_loc.image_path,
                    context_info
                )
                return description
            except Exception as e:
                logger.warning(f"Vision API description failed: {e}")
        
        # Fallback: basic description
        return self._generate_basic_description(image_loc, context_info)
    
    def _get_vision_prompt(self, context_above: str, context_below: str) -> str:
        """
        Generate RAGFlow-style prompt for image analysis with context.
        Supports both structured data (tables/charts) and general figures.
        """
        prompt = """## ROLE
You are an expert visual data analyst with deep expertise in extracting and interpreting visual information.

## GOAL
Analyze the image and produce a textual representation strictly based on what is visible.
Surrounding context may be used only for minimal clarification or disambiguation of terms that appear in the image, not as a source of new information.
"""
        
        if context_above:
            prompt += f"\n## CONTEXT (ABOVE)\n\n{context_above}\n"
        
        if context_below:
            prompt += f"\n## CONTEXT (BELOW)\n\n{context_below}\n"
        
        prompt += """
## DECISION RULE (CRITICAL)

First, determine whether the image contains an explicit visual data representation with enumerable data units forming a coherent dataset.

Enumerable data units are clearly separable, repeatable elements intended for comparison, measurement, or aggregation, such as:
- Rows or columns in a table
- Individual bars in a bar chart
- Identifiable data points or series in a line graph
- Labeled segments in a pie chart
- UI table rows with repeated structure

The mere presence of numbers, icons, UI elements, or labels does NOT qualify unless they together form such a dataset.

## TASKS

1. Inspect the image and determine which output mode applies based on the decision rule.
2. Use surrounding context only to disambiguate terms that appear in the image.
3. Follow the output rules strictly.
4. Include only content that is explicitly visible in the image.
5. Do not infer intent, functionality, process logic, or meaning beyond what is visually or textually shown.

## OUTPUT RULES (STRICT)

- Produce output in exactly one of the two modes defined below.
- Do NOT mention, label, or reference the modes in the output.
- Do NOT combine content from both modes.
- Do NOT explain or justify the choice of mode.
- Do NOT add any headings, titles, or commentary beyond what the mode requires.

---

## MODE 1: STRUCTURED VISUAL DATA OUTPUT

(Use only if the image contains enumerable data units forming a coherent dataset.)

Output only the following fields:
- Visual Type: (e.g., table, bar chart, line chart, pie chart, etc.)
- Title: (if visible)
- Axes / Legends / Labels: (list key labels and their meanings)
- Data Points: (concise list of key data rows or values)
- Captions / Annotations: (any visible annotations or notes)

---

## MODE 2: GENERAL FIGURE CONTENT

(Use only if the image does NOT contain enumerable data units.)

Write the content directly, starting from the first sentence. Do NOT add any introductory labels, titles, headings, or prefixes.

Requirements:
- Describe visible regions and components in a stable order (e.g., top-to-bottom, left-to-right).
- Explicitly name interface elements or visual objects exactly as they appear (e.g., tabs, panels, buttons, icons, input fields).
- Transcribe all visible text verbatim; do not paraphrase, summarize, or reinterpret labels.
- Describe spatial grouping, containment, and alignment of elements.
- Do NOT interpret intent, behavior, workflows, or processes.
- Avoid narrative or stylistic language unless it is a dominant and functional visual element.

Use concise, information-dense sentences.
"""
        
        return prompt
    
    def _describe_image_with_vision(
        self,
        image_path: str,
        context_info: Dict[str, str]
    ) -> str:
        """Use Vision API to describe image with advanced RAGFlow-style prompting"""
        try:
            # Determine client type based on provider
            if self.vision_provider == "gemini":
                return self._describe_image_with_gemini(image_path, context_info)
            else:
                # Default to OpenAI-compatible client
                return self._describe_image_with_openai(image_path, context_info)
        except Exception as e:
            logger.warning(f"Vision API description failed: {e}")
            raise
    
    def _describe_image_with_openai(
        self,
        image_path: str,
        context_info: Dict[str, str]
    ) -> str:
        """Use OpenAI Vision API to describe image"""
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        suffix = Path(image_path).suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        
        # Build context-aware prompt
        context_above = context_info.get("context_above", "")
        context_below = context_info.get("context_below", "")
        prompt = self._get_vision_prompt(context_above, context_below)
        
        response = self.vision_client.chat.completions.create(
            model=self.vision_model,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
        )
        
        description = response.choices[0].message.content or ""
        if isinstance(description, list):
            description = " ".join(
                part.text for part in description if getattr(part, "text", None)
            )
        
        logger.debug("Generated OpenAI Vision description for image")
        return str(description)
    
    def _describe_image_with_gemini(
        self,
        image_path: str,
        context_info: Dict[str, str]
    ) -> str:
        """Use Google Gemini Vision API to describe image"""
        try:
            import mimetypes
            import google.genai as genai
            from google.genai import types
        except ImportError:
            logger.warning("google-generativeai package not installed. Falling back to basic description.")
            return self._generate_basic_description(ImageLocation(image_path=image_path, page_num=0, x0=0, y0=0, x1=0, y1=0, width=0, height=0), context_info)

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

        # Build context-aware prompt
        context_above = context_info.get("context_above", "")
        context_below = context_info.get("context_below", "")
        prompt = self._get_vision_prompt(context_above, context_below)

        # Create image part
        image_part = types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type))

        # Generate content using Gemini API
        response = self.vision_client.models.generate_content(
            model=f'{self.vision_model}',
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                max_output_tokens=1500,
                temperature=0.7,
            )
        )

        description = response.text or ""
        logger.debug("Generated Gemini Vision description for image")
        return str(description)
    
    def _calculate_bbox_overlap_ratio(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate the overlap ratio between two bounding boxes (table and image).
        
        Args:
            bbox1: (x0, y0, x1, y1) - table bounding box
            bbox2: (x0, y0, x1, y1) - image bounding box
        
        Returns:
            Overlap ratio (0.0 to 1.0)
            Returns the maximum of:
            - intersection / table_area (what % of table overlaps with image)
            - intersection / image_area (what % of image overlaps with table)
            Both directions matter: if table is mostly in image OR image is mostly in table
        """
        x0_1, y0_1, x1_1, y1_1 = bbox1  # table
        x0_2, y0_2, x1_2, y1_2 = bbox2  # image
        
        # Calculate intersection
        inter_x0 = max(x0_1, x0_2)
        inter_y0 = max(y0_1, y0_2)
        inter_x1 = min(x1_1, x1_2)
        inter_y1 = min(y1_1, y1_2)
        
        # No intersection
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            return 0.0
        
        # Calculate areas
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        table_area = (x1_1 - x0_1) * (y1_1 - y0_1)
        image_area = (x1_2 - x0_2) * (y1_2 - y0_2)
        
        if table_area < 1 or image_area < 1:
            return 0.0
        
        # Return the maximum ratio from both directions
        # This catches cases where:
        # 1. Table is mostly inside image (high table_overlap)
        # 2. Image is mostly inside table (high image_overlap)
        table_overlap = inter_area / table_area
        image_overlap = inter_area / image_area
        
        return max(table_overlap, image_overlap)
    
    def _generate_basic_description(
        self,
        image_loc: ImageLocation,
        context_info: Dict[str, str]
    ) -> str:
        """Generate basic image description without Vision API"""
        path = Path(image_loc.image_path)
        
        desc = f"Image: {path.name} "
        desc += f"(Position: page {image_loc.page_num}, "
        desc += f"Size: {image_loc.width:.0f}x{image_loc.height:.0f}px)"
        
        # Include both context_above and context_below
        context_above = context_info.get("context_above", "")
        context_below = context_info.get("context_below", "")
        
        if context_above or context_below:
            desc += "\n\n## Surrounding Context"
            if context_above:
                desc += f"\n\nBefore image: {context_above[:150]}..."
            if context_below:
                desc += f"\n\nAfter image: {context_below[:150]}..."
        
        return desc
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format table as text"""
        if not table:
            return ""
        
        # Convert to markdown-like format
        lines = []
        for row in table:
            lines.append(" | ".join(str(cell) if cell else "" for cell in row))
        
        return "\n".join(lines)

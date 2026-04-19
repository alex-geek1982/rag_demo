"""
RAG Engine - Advanced chunking strategies inspired by RAGFlow

Implements two complementary chunking strategies:
1. TitleChunker: For documents with paragraph/section structure (PDF, MD, DOCX)
   - Detects heading levels and maintains document hierarchy
   - Preserves semantic boundaries
   
2. TokenChunker: For unstructured text or plain text documents
   - Splits by token count with configurable size and overlap
   - Supports custom delimiters for sentence-level splitting
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..types import ContentBlock, ContentType, Chunk
from ..config import ProcessingConfig

logger = logging.getLogger(__name__)


class TitleLevel(Enum):
    """Document title levels"""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


class TitleChunker:
    """
    Chunks documents with paragraph/section structure by auto-detecting hierarchy.
    
    Now supports markdown-aware chunking for converted markdown content:
    - Directly parses markdown headings (# ## ###) for perfect hierarchy detection
    - Falls back to auto-detection for non-markdown content
    - Preserves semantic structure through hierarchical grouping
    
    Supports hierarchical + token-based chunking inspired by RAGFlow:
    - Assigns section IDs based on heading levels
    - Merges chunks within sections based on token limits
    - Maintains semantic boundaries
    
    Hierarchy Detection Strategies:
    1. Markdown parsing: Direct parsing of # ## ### headings (highest priority)
    2. Outline matching: Extract PDF outlines if available, match with high similarity (>0.8)
    3. Pattern hints + frequency matching: Analyze text patterns and layout to infer hierarchy
    4. Layout matching: Use block metadata (layout_type, layoutno)
    """
    
    BODY_LEVEL = 10000  # Level assigned to body text (not a title)
    DEFAULT_FALLBACK_LEVEL = 2  # Default level for text marked as potential title
    
    def __init__(self, include_heading_content: bool = False, use_outline: bool = True, chunk_token_size: int = 512):
        """
        Initialize TitleChunker with auto-hierarchy detection
        
        Args:
            include_heading_content: Whether to treat heading text as separate chunks
            use_outline: Whether to attempt PDF outline extraction (if available)
            chunk_token_size: Maximum token count per chunk (approximate)
        """
        self.include_heading_content = include_heading_content
        self.use_outline = use_outline
        self.chunk_token_size = max(1, chunk_token_size)
        
        # Initialize TokenChunker for fallback
        self.token_chunker = TokenChunker(chunk_token_size=self.chunk_token_size)
        
        # Regex patterns for common title formats (Markdown, numbered, etc.)
        # These are hints, NOT strict definitions
        self.heading_hint_patterns = [
            (r"^#{1,6}\s+\S", 'markdown'),        # Markdown: # to ######
            (r"^\d+\.\s+\S", 'numbered'),         # Numbered: 1. Section
            (r"^[A-Z][A-Z\s]{2,}$", 'uppercase'),  # ALL CAPS
            (r"^\s{0,4}[A-Z][\w\s]{0,50}[:\.]?$", 'title_case'),  # Title Case - short titles only
        ]
    
    @staticmethod
    def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
        """
        Calculate character-level similarity for outline matching
        
        Args:
            a, b: Strings to compare
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if similarity > threshold
        """
        # Simple character-level similarity (RAGFlow-style)
        a_lower, b_lower = a.lower().strip(), b.lower().strip()
        
        # Early exit for very different lengths
        if len(a_lower) == 0 or len(b_lower) == 0:
            return False
        
        matching = sum(1 for x, y in zip(a_lower, b_lower) if x == y)
        similarity = matching / max(len(a_lower), len(b_lower))
        
        return similarity > threshold
    
    def _is_markdown_content(self, blocks: List[ContentBlock]) -> bool:
        """
        Check if the content blocks contain converted markdown content.
        
        Returns:
            True if this appears to be markdown content from document processor
        """
        if not blocks:
            return False
        
        # Check metadata for markdown conversion marker
        for block in blocks:
            if block.metadata and block.metadata.get("source") == "converted_markdown":
                return True
        
        # Fallback: check if content contains markdown heading patterns
        if len(blocks) == 1 and blocks[0].type == ContentType.TEXT:
            content = blocks[0].content
            # Look for markdown headings in the content
            if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
                return True
        
        return False
    
    def _chunk_markdown_content(self, blocks: List[ContentBlock]) -> List[Chunk]:
        """
        Chunk markdown content by parsing markdown headings directly.
        
        This provides perfect hierarchy detection for converted markdown content.
        """
        if not blocks or len(blocks) != 1:
            return []
        
        block = blocks[0]
        content = block.content
        
        # Split content into lines
        lines = content.split('\n')
        
        # Parse markdown structure
        markdown_structure = self._parse_markdown_structure(lines)
        
        # Build chunks from parsed structure
        chunks = self._build_chunks_from_markdown_structure(markdown_structure, block.id)
        
        logger.info(f"TitleChunker: Generated {len(chunks)} chunks from markdown content")
        return chunks
    
    def _parse_markdown_structure(self, lines: List[str]) -> List[Tuple[str, int, int, str]]:
        """
        Parse markdown content and extract structure with heading levels, bullet points, and tables.
        
        Returns:
            List of tuples: (line_content, heading_level, line_index, content_type)
            heading_level is BODY_LEVEL for non-heading lines
            content_type: 'heading', 'bullet', 'table', 'text', 'table_sep'
        """
        structure = []
        in_table = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check for markdown heading
            heading_match = re.match(r'^(#{1,6})\s*(.+)', stripped)
            if heading_match:
                in_table = False
                level = len(heading_match.group(1))
                # Preserve original line content with '##' symbols intact
                structure.append((stripped, level, i, 'heading'))
                continue
            
            # Check for table separator line (|---|)
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                structure.append((stripped, 100, i, 'table_sep'))
                in_table = True
                continue
            
            # Check for table row (contains |)
            if '|' in stripped and (in_table or self._looks_like_table_row(stripped)):
                structure.append((stripped, 100, i, 'table'))
                in_table = True
                continue
            
            # Check for bullet points (numbered, parentheses, chinese, dashes, asterisks)
            bullet_match = re.match(r'^([\s\t]*)(\d+[\.\)、]|[\-\*\+])\s+(.+)', stripped)
            if bullet_match:
                in_table = False
                indent = len(bullet_match.group(1))
                # Bullet level based on indentation (0-3 levels)
                bullet_level = min(3, indent // 2)
                # Use level 10 + bullet_level to distinguish bullets from headings
                structure.append((stripped, 10 + bullet_level, i, 'bullet'))
                continue
            
            in_table = False
            # Regular content line
            structure.append((stripped, self.BODY_LEVEL, i, 'text'))
        
        return structure
    
    def _looks_like_table_row(self, line: str) -> bool:
        """
        Check if a line looks like a table row.
        
        Heuristics:
        - Has at least 2 pipe separators
        - Content between pipes is relatively uniform
        """
        if line.count('|') < 2:
            return False
        
        # Should have content between pipes
        parts = line.split('|')
        non_empty = [p for p in parts if p.strip()]
        
        # At least 2 non-empty parts
        return len(non_empty) >= 2
    
    def _build_chunks_from_markdown_structure(
        self, 
        structure: List[Tuple[str, int, int, str]], 
        source_block_id: str
    ) -> List[Chunk]:
        """
        Build chunks from parsed markdown structure using a manual-like section merge strategy.

        This method follows the RAGFlow manual chunking logic for markdown:
        - Headings define section boundaries based on the most important title level
        - Tables are treated as atomic blocks and preserved whole
        - Small chunks (< 32 tokens) are merged across section boundaries
        - Chunks in the same section can grow up to 1024 tokens before splitting
        - Tables can merge into the previous chunk while still respecting limits
        """
        if not structure:
            return []
        
        contents = []
        levels = []
        content_types = []
        line_indices = []
        
        for item in structure:
            content, level, idx, content_type = item
            contents.append(content)
            levels.append(level)
            content_types.append(content_type)
            line_indices.append(idx)
        
        sec_ids = self._assign_section_ids_enhanced(levels, content_types)
        sections = [
            (contents[i], sec_ids[i], line_indices[i], content_types[i])
            for i in range(len(contents))
        ]
        sections.sort(key=lambda x: x[2])

        grouped_sections = []
        i = 0
        while i < len(sections):
            txt, sec_id, line_idx, content_type = sections[i]
            if content_type in {'table', 'table_sep'}:
                table_lines = [txt]
                table_line_idx = line_idx
                j = i + 1
                while j < len(sections) and sections[j][3] in {'table', 'table_sep'}:
                    table_lines.append(sections[j][0])
                    j += 1
                grouped_sections.append(("\n".join(table_lines), -1, table_line_idx, 'table'))
                i = j
            else:
                grouped_sections.append((txt, sec_id, line_idx, content_type))
                i += 1
        sections = grouped_sections

        chunks = []
        last_sid = -2
        tk_cnt = 0
        
        for txt, sec_id, _, content_type in sections:
            token_count = self._count_tokens(txt)
            
            if not chunks:
                chunk = Chunk(
                    text=txt,
                    chunk_type='text',
                    source_block_ids=[source_block_id],
                    metadata={
                        'chunking_method': 'manual_markdown',
                        'section_id': sec_id,
                        'token_count': token_count,
                        'content_type': content_type
                    }
                )
                chunks.append(chunk)
                tk_cnt = token_count
                if sec_id > -1:
                    last_sid = sec_id
                continue

            if sec_id != last_sid and sec_id != -1:
                chunk = Chunk(
                    text=txt,
                    chunk_type='text',
                    source_block_ids=[source_block_id],
                    metadata={
                        'chunking_method': 'manual_markdown',
                        'section_id': sec_id,
                        'token_count': token_count,
                        'content_type': content_type
                    }
                )
                chunks.append(chunk)
                tk_cnt = token_count
                if sec_id > -1:
                    last_sid = sec_id
            elif tk_cnt < 32 or (tk_cnt < 1024 and (sec_id == last_sid or sec_id == -1)):
                chunks[-1].text += "\n" + txt
                chunks[-1].source_block_ids.append(source_block_id)
                tk_cnt += token_count
                chunks[-1].metadata['token_count'] = tk_cnt
            else:
                chunk = Chunk(
                    text=txt,
                    chunk_type='text',
                    source_block_ids=[source_block_id],
                    metadata={
                        'chunking_method': 'manual_markdown',
                        'section_id': sec_id,
                        'token_count': token_count,
                        'content_type': content_type
                    }
                )
                chunks.append(chunk)
                tk_cnt = token_count
                if sec_id > -1:
                    last_sid = sec_id

        for chunk in chunks:
            chunk.metadata['token_count'] = self._count_tokens(chunk.text)

        logger.debug(f"Generated {len(chunks)} manual-style markdown chunks")
        return chunks
    
    def _assign_section_ids_enhanced(self, levels: List[int], content_types: List[str]) -> List[int]:
        """
        Assign section IDs based on document structure.
        
        Rules:
        - Determine section level: if level 1 count > 1, use level 1; else use level 2
        - Only headings at the section level start new sections
        - Tables and other content are assigned section -1
        - Body text and bullets inherit the current section
        """
        # Count level occurrences
        level_counts = {}
        for level in levels:
            if level != self.BODY_LEVEL:
                level_counts[level] = level_counts.get(level, 0) + 1
        
        # Determine section level
        section_level = 1 if level_counts.get(1, 0) > 1 else 2
        
        sec_ids = []
        sid = -1
        
        for level, content_type in zip(levels, content_types):
            if content_type == 'heading' and level == section_level:
                sid += 1
                sec_ids.append(sid)
            elif content_type in {'table', 'table_sep'}:
                sec_ids.append(-1)
            else:
                # Body text, bullets, or headings not at section level
                sec_ids.append(sid if sid >= 0 else -1)
        
        return sec_ids
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation: ~4 chars per token)"""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def chunk(self, blocks: List[ContentBlock]) -> List[Chunk]:
        """
        Chunk blocks by auto-detected hierarchy or token size
        
        Automatically selects chunking strategy:
        - Use TitleChunker if document is markdown and has multiple levels or multiple level 1 headings
        - Otherwise use TokenChunker
        
        Args:
            blocks: List of content blocks to chunk
        
        Returns:
            List of chunks
        """
        if not blocks:
            return []
        
        # Check if this is markdown content
        if self._is_markdown_content(blocks):
            # Parse markdown structure to check conditions
            if len(blocks) == 1 and blocks[0].type == ContentType.TEXT:
                content = blocks[0].content
                lines = content.split('\n')
                structure = self._parse_markdown_structure(lines)
                
                # Count levels
                level_counts = {}
                for _, level, _, _ in structure:
                    if level != self.BODY_LEVEL:
                        level_counts[level] = level_counts.get(level, 0) + 1
                
                num_levels = len(level_counts)
                num_level1 = level_counts.get(1, 0)
                
                # Use TitleChunker if multiple levels or multiple level 1 headings
                if num_levels > 1 or num_level1 > 1:
                    logger.debug("Using TitleChunker for markdown with hierarchical structure")
                    return self._chunk_markdown_content(blocks)
                else:
                    logger.debug("Using TokenChunker for simple markdown content")
                    return self.token_chunker.chunk(blocks)
        
        # For non-markdown content, use TokenChunker
        logger.debug("Using TokenChunker for non-markdown content")
        return self.token_chunker.chunk(blocks)
    
    def _detect_title_levels(
        self,
        line_records: List[Tuple[ContentBlock, str]],
        blocks: List[ContentBlock]
    ) -> List[int]:
        """
        Auto-detect title hierarchy levels using RAGFlow's strategies.
        
        Strategies (in order):
        1. Outline matching: Try to extract and match PDF outline
        2. Regex pattern hints: Detect common title patterns
        3. Frequency matching: Analyze which level is most likely
        4. Layout matching: Use block metadata (layout_type, layoutno)
        
        Returns:
            List of detected levels (BODY_LEVEL for non-titles)
        """
        levels = []
        
        # Strategy 1: Try outline matching if available
        outline_matches = self._try_outline_matching(line_records, blocks)
        if outline_matches and sum(1 for l in outline_matches if l != self.BODY_LEVEL) > len(line_records) * 0.03:
            logger.debug("Using outline matching for hierarchy detection")
            return outline_matches
        
        # Strategy 2 & 3: Pattern hints + frequency matching
        pattern_levels = self._detect_by_pattern_hints(line_records)
        
        # If pattern hints found any titles, use them; otherwise use frequency
        if any(l != self.BODY_LEVEL for l in pattern_levels):
            logger.debug("Using pattern hint matching for hierarchy")
            levels = pattern_levels
        else:
            logger.debug("Using frequency matching for hierarchy")
            levels = self._detect_by_frequency(line_records)
        
        # Strategy 4: Fallback - layout matching
        levels = self._apply_layout_matching(levels, line_records)
        
        return levels
    
    def _try_outline_matching(
        self,
        line_records: List[Tuple[ContentBlock, str]],
        blocks: List[ContentBlock]
    ) -> Optional[List[int]]:
        """
        Try to match document with PDF outline (if available).
        
        Returns:
            List of levels if outline matching successful, None otherwise
        """
        # Try to extract outlines from blocks' metadata
        outlines = []
        for block in blocks:
            if block.metadata and 'outline' in block.metadata:
                outline_item = block.metadata['outline']
                if isinstance(outline_item, dict):
                    outlines.append((outline_item.get('title', ''), outline_item.get('level', 1)))
        
        if not outlines:
            return None
        
        # Try to match each line with outline items
        levels = []
        for block, line in line_records:
            matched = False
            for outline_title, outline_level in outlines:
                if self._similar(line, outline_title, threshold=0.8):
                    levels.append(outline_level)
                    matched = True
                    break
            
            if not matched:
                levels.append(self.BODY_LEVEL)
        
        return levels if levels else None
    
    def _detect_by_pattern_hints(self, line_records: List[Tuple[ContentBlock, str]]) -> List[int]:
        """
        Detect titles using regex pattern hints.
        
        Only markdown and numbered patterns are reliable for title detection.
        Title case pattern is unreliable and skipped to avoid false positives.
        
        Returns:
            List of levels (BODY_LEVEL for non-titles)
        """
        levels = []
        detected_patterns = {}  # Track which patterns match
        
        for block, line in line_records:
            line_level = self.BODY_LEVEL
            stripped_line = line.strip()
            
            # Only use reliable patterns (markdown and numbered)
            # Skip title_case to avoid false positives
            reliable_patterns = [
                (r"^#{1,6}\s+\S", 'markdown'),
                (r"^\d+\.\s+\S", 'numbered'),
                (r"^[A-Z][A-Z\s]{2,}$", 'uppercase'),
            ]
            
            for pattern, pattern_name in reliable_patterns:
                if re.match(pattern, stripped_line):
                    # Pattern matched - but we need to infer the actual level
                    if pattern_name == 'markdown':
                        # For markdown, extract level from #'s
                        hash_count = len(re.match(r"^#+", stripped_line).group())
                        line_level = hash_count
                    else:
                        # For other patterns, assign based on frequency
                        line_level = detected_patterns.get(pattern_name, len(detected_patterns) + 1)
                        detected_patterns[pattern_name] = line_level
                    break
            
            levels.append(line_level)
        
        return levels
    
    def _detect_by_frequency(self, line_records: List[Tuple[ContentBlock, str]]) -> List[int]:
        """
        Detect titles by analyzing text characteristics and frequency.
        
        Heuristics:
        - Shorter lines are more likely titles
        - Lines followed by longer blocks are likely titles
        - Lines with consistent formatting are likely titles
        
        Returns:
            List of levels
        """
        levels = [self.BODY_LEVEL] * len(line_records)
        
        if len(line_records) < 3:
            return levels
        
        # Calculate line lengths
        line_lengths = [len(line) for _, line in line_records]
        avg_length = sum(line_lengths) / len(line_lengths)
        short_threshold = avg_length * 0.5  # Lines 50% shorter than avg are likely titles
        
        # Mark potentially short lines as titles (level 1 by default)
        for i, (_, line) in enumerate(line_records):
            if len(line.strip()) > 0 and len(line) < short_threshold:
                # Additional check: if followed by much longer content, likely a title
                if i + 1 < len(line_records):
                    next_length = line_lengths[i + 1]
                    if next_length > len(line) * 1.5:
                        levels[i] = 1
        
        return levels
    
    def _apply_layout_matching(
        self,
        levels: List[int],
        line_records: List[Tuple[ContentBlock, str]]
    ) -> List[int]:
        """
        Apply layout metadata matching to refine hierarchy.
        
        Checks block metadata for layout_type and layoutno info.
        Assigns levels based on keywords: 'title', 'section', 'head', etc.
        """
        for i, (block, line) in enumerate(line_records):
            if levels[i] != self.BODY_LEVEL:
                continue  # Already detected as title
            
            metadata = block.metadata or {}
            layout_type = metadata.get('layout_type', '').lower()
            
            # Check for layout keywords
            if any(kw in layout_type for kw in ['title', 'section', 'head', 'chapter']):
                # Assign level based on layoutno if available
                layout_no = metadata.get('layoutno', 0)
                if layout_no > 0:
                    levels[i] = layout_no
                else:
                    # Use fallback level for identified but unranked titles
                    if not self._is_short_line(line):
                        levels[i] = self.DEFAULT_FALLBACK_LEVEL
        
        return levels
    
    @staticmethod
    def _is_short_line(line: str, max_length: int = 100) -> bool:
        """Check if line is short enough to be a title"""
        return len(line.strip()) < max_length
    
    def _extract_line_records(self, blocks: List[ContentBlock]) -> List[Tuple[ContentBlock, str]]:
        """
        Extract text lines from content blocks with intelligent line merging.
        
        Handles large line spacing by merging lines that are part of the same sentence:
        - If a line doesn't end with punctuation, merge it with the next line
        - Respects title patterns to avoid merging across titles
        """
        records = []
        
        for block in blocks:
            if block.type == ContentType.TEXT:
                # Split text block into lines
                lines = block.content.split('\n')
                merged_lines = self._merge_incomplete_lines(lines)
                
                for line in merged_lines:
                    if line.strip():  # Skip empty lines
                        records.append((block, line))
            elif block.type == ContentType.IMAGE:
                # Treat images as non-text content
                records.append((block, f"[Image: {block.metadata.get('description', 'image') if block.metadata else 'image'}]"))
            elif block.type == ContentType.TABLE:
                # Treat tables as non-text content
                records.append((block, f"[Table: {block.content[:100] if block.content else 'table'}...]"))
        
        return records
    
    def _merge_incomplete_lines(self, lines: List[str]) -> List[str]:
        """
        Merge lines that are incomplete (don't end with punctuation) with the next line.
        
        This handles cases where large line spacing in documents causes a single sentence
        to be split across multiple lines.
        
        Args:
            lines: List of text lines from a block
            
        Returns:
            List of merged lines
        """
        if not lines:
            return []
        
        # Chinese and common punctuation marks that indicate sentence end
        sentence_endings = {'。', '！', '？', '!', '?', '；', ';', '，', ',', '：', ':', '\n'}
        
        merged = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Skip empty lines
            if not current_line:
                i += 1
                continue
            
            # Check if this line is incomplete (doesn't end with punctuation)
            should_merge = (
                current_line and 
                not any(current_line.endswith(p) for p in sentence_endings) and
                current_line[-1].isalnum()  # Ends with alphanumeric character
            )
            
            # Look ahead to merge with next lines if needed
            if should_merge and i + 1 < len(lines):
                # Check if next line is a title (should not merge)
                next_line = lines[i + 1].strip()
                
                if next_line and not self._is_likely_title(next_line):
                    # Merge with next line
                    merged_text = current_line + ' ' + next_line
                    i += 2
                    
                    # Continue merging if the merged line is still incomplete
                    while (i < len(lines) and 
                           not any(merged_text.rstrip().endswith(p) for p in sentence_endings) and
                           merged_text.rstrip()[-1].isalnum() if merged_text.rstrip() else False):
                        next_line = lines[i].strip()
                        if next_line and not self._is_likely_title(next_line):
                            merged_text = merged_text + ' ' + next_line
                            i += 1
                        else:
                            break
                    
                    merged.append(merged_text)
                else:
                    merged.append(current_line)
                    i += 1
            else:
                merged.append(current_line)
                i += 1
        
        return merged
    
    def _is_likely_title(self, line: str) -> bool:
        """
        Check if a line is likely to be a title/heading.
        
        Used to prevent merging lines that are separated by a title.
        """
        if not line:
            return False
        
        stripped = line.strip()
        
        # Check against known title patterns
        for pattern, _ in self.heading_hint_patterns:
            if re.match(pattern, stripped):
                return True
        
        # Check if it's very short AND all uppercase (likely a title)
        if len(stripped) < 20 and stripped.isupper() and len(stripped) > 1:
            return True
        
        return False

    def _build_chunks_from_hierarchy(
        self,
        line_records: List[Tuple[ContentBlock, str]],
        levels: List[int]
    ) -> List[Chunk]:
        """Build chunks following auto-detected hierarchy"""
        chunks = []
        current_group = []
        current_titles = []  # Stack of current title context
        
        for (block, line), level in zip(line_records, levels):
            if level != self.BODY_LEVEL:  # This is a detected title
                # Flush current group before starting new section
                if current_group and any(text.strip() for _, text in current_group):
                    chunks.extend(self._create_chunks_from_group(
                        current_group, current_titles
                    ))
                    current_group = []
                
                # Update title context
                # Remove titles at same or deeper level
                while current_titles and current_titles[-1][1] >= level:
                    current_titles.pop()
                
                current_titles.append((line, level))
                
                # Optionally create chunk for heading itself
                if self.include_heading_content:
                    chunk = Chunk(
                        text=line,
                        chunk_type='title',
                        source_block_ids=[block.id],
                        title=' > '.join(title for title, _ in current_titles),
                        title_level=level
                    )
                    chunks.append(chunk)
            else:
                # Body text
                current_group.append((block, line))
        
        # Flush final group
        if current_group and any(text.strip() for _, text in current_group):
            chunks.extend(self._create_chunks_from_group(current_group, current_titles))
        
        return chunks
    
    def _create_chunks_from_group(
        self,
        group: List[Tuple[ContentBlock, str]],
        title_context: List[Tuple[str, int]]
    ) -> List[Chunk]:
        """Create one or more chunks from a group of lines using token-aware splitting."""
        if not group:
            return []
        
        def build_chunk(lines: List[Tuple[ContentBlock, str]]) -> Chunk:
            text_parts = [line for _, line in lines]
            combined_text = '\n'.join(text_parts)
            source_ids = list(set(block.id for block, _ in lines))
            title_str = ' > '.join(title for title, _ in title_context) if title_context else ""
            
            return Chunk(
                text=combined_text,
                chunk_type='text',
                source_block_ids=source_ids,
                metadata={
                    'token_count': self._count_tokens(combined_text),
                    'content_type': 'text'
                },
                title=title_str,
                title_level=title_context[-1][1] if title_context else 0
            )
        
        chunks: List[Chunk] = []
        current_lines: List[Tuple[ContentBlock, str]] = []
        current_token_count = 0
        
        for block, line in group:
            token_count = self._count_tokens(line)
            if current_lines and current_token_count + token_count > self.chunk_token_size:
                chunks.append(build_chunk(current_lines))
                current_lines = []
                current_token_count = 0
            
            current_lines.append((block, line))
            current_token_count += token_count
        
        if current_lines:
            chunks.append(build_chunk(current_lines))
        
        return chunks



class TokenChunker:
    """
    Chunks documents by token count with configurable overlap.
    
    Follows RAGFlow's token_chunker approach:
    - Splits text by configurable token size
    - Supports custom delimiters (sentence, paragraph, etc.)
    - Maintains overlap between chunks
    - Handles special content types (table, image) separately
    """
    
    def __init__(
        self,
        chunk_token_size: int = 512,
        overlapped_percent: float = 0,
        delimiters: Optional[List[str]] = None,
        table_context_size: int = 0,
        image_context_size: int = 0
    ):
        """
        Initialize TokenChunker
        
        Args:
            chunk_token_size: Target token count per chunk (approximate)
            overlapped_percent: Overlap percentage (0-100)
            delimiters: List of delimiters to split on (e.g., ['\n', '。', '.'])
            table_context_size: Context tokens to add around tables
            image_context_size: Context tokens to add around images
        """
        self.chunk_token_size = max(1, chunk_token_size)
        self.overlapped_percent = max(0, min(overlapped_percent, 100))
        self.delimiters = delimiters or ['\n']
        self.table_context_size = max(0, table_context_size)
        self.image_context_size = max(0, image_context_size)
    
    def chunk(self, blocks: List[ContentBlock]) -> List[Chunk]:
        """
        Chunk blocks by token size
        
        Args:
            blocks: List of content blocks to chunk
        
        Returns:
            List of token-sized chunks
        """
        if not blocks:
            return []
        
        # Build internal chunk format
        internal_chunks = self._build_internal_chunks(blocks)
        if not internal_chunks:
            return []
        
        # Merge text chunks by token size
        merged_chunks = self._merge_text_chunks_by_token_size(internal_chunks)
        
        # Attach context to media chunks
        if self.table_context_size > 0 or self.image_context_size > 0:
            merged_chunks = self._attach_context_to_media_chunks(merged_chunks)
        
        # Convert to final Chunk objects
        result_chunks = self._finalize_chunks(merged_chunks)
        
        logger.info(f"TokenChunker: Generated {len(result_chunks)} chunks from {len(blocks)} blocks")
        return result_chunks
    
    def _build_internal_chunks(self, blocks: List[ContentBlock]) -> List[Dict[str, Any]]:
        """Convert content blocks to internal chunk format"""
        chunks = []
        delimiter_pattern = self._compile_delimiter_pattern(self.delimiters)
        
        for block in blocks:
            if block.type == ContentType.TEXT:
                # Split text by delimiters
                text_segments = self._split_text_by_pattern(block.content, delimiter_pattern)
                
                for segment in text_segments:
                    if segment.strip():
                        chunks.append({
                            'text': segment,
                            'type': 'text',
                            'source_block_id': block.id,
                            'token_count': self._count_tokens(segment),
                            'content_type': 'text'
                        })
            
            elif block.type == ContentType.TABLE:
                chunks.append({
                    'text': block.content,
                    'type': 'table',
                    'source_block_id': block.id,
                    'token_count': self._count_tokens(block.content),
                    'content_type': 'table',
                    'context_above': '',
                    'context_below': ''
                })
            
            elif block.type == ContentType.IMAGE:
                description = block.metadata.get('description', '') if block.metadata else ''
                chunks.append({
                    'text': description,
                    'type': 'image',
                    'source_block_id': block.id,
                    'token_count': self._count_tokens(description),
                    'content_type': 'image',
                    'context_above': '',
                    'context_below': ''
                })
        
        return chunks
    
    def _compile_delimiter_pattern(self, delimiters: List[str]) -> str:
        """Compile delimiters into regex pattern"""
        if not delimiters:
            return ""
        
        # Sort by length (longest first) to match longest delimiters first
        sorted_delimiters = sorted(set(delimiters), key=len, reverse=True)
        escaped = [re.escape(d) for d in sorted_delimiters]
        return "|".join(escaped)
    
    def _split_text_by_pattern(self, text: str, pattern: str) -> List[str]:
        """Split text by delimiter pattern while keeping delimiters"""
        if not pattern:
            return [text] if text else []
        
        # Split and keep delimiters
        split_texts = re.split(f"({pattern})", text, flags=re.DOTALL)
        
        chunks = []
        for i in range(0, len(split_texts), 2):
            chunk = split_texts[i]
            if chunk:  # Keep non-empty parts
                if i + 1 < len(split_texts):
                    chunk += split_texts[i + 1]  # Append delimiter
                if chunk.strip():
                    chunks.append(chunk)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation: ~4 chars per token)"""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def _merge_text_chunks_by_token_size(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge text chunks to reach target token size"""
        merged = []
        prev_text_idx = -1
        threshold = self.chunk_token_size * (100 - self.overlapped_percent) / 100.0
        
        for chunk in chunks:
            if chunk['type'] != 'text':
                merged.append(chunk.copy())
                prev_text_idx = -1
                continue
            
            current = chunk.copy()
            should_start_new = prev_text_idx < 0 or merged[prev_text_idx]['token_count'] > threshold
            
            if should_start_new:
                # Add overlap from previous chunk if needed
                if (prev_text_idx >= 0 and self.overlapped_percent > 0 and 
                    merged[prev_text_idx]['text']):
                    overlap_start = int(
                        len(merged[prev_text_idx]['text']) * 
                        (100 - self.overlapped_percent) / 100.0
                    )
                    current['text'] = (
                        merged[prev_text_idx]['text'][overlap_start:] + 
                        current['text']
                    )
                    current['token_count'] = self._count_tokens(current['text'])
                
                merged.append(current)
                prev_text_idx = len(merged) - 1
            else:
                # Merge with previous chunk
                merged[prev_text_idx]['text'] += '\n' + current['text']
                merged[prev_text_idx]['token_count'] += current['token_count']
                merged[prev_text_idx]['source_block_id'] = (
                    merged[prev_text_idx]['source_block_id'] + ',' + current['source_block_id']
                )
        
        return merged
    
    def _attach_context_to_media_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attach surrounding text context to table/image chunks"""
        for i, chunk in enumerate(chunks):
            if chunk['type'] not in {'table', 'image'}:
                continue
            
            context_size = (
                self.image_context_size if chunk['type'] == 'image' 
                else self.table_context_size
            )
            if context_size <= 0:
                continue
            
            # Collect context above
            remain_above = context_size
            parts_above = []
            
            prev_idx = i - 1
            while prev_idx >= 0 and remain_above > 0:
                prev_chunk = chunks[prev_idx]
                if prev_chunk['type'] == 'text':
                    if prev_chunk['token_count'] >= remain_above:
                        parts_above.insert(
                            0, 
                            self._take_sentences(
                                prev_chunk['text'], 
                                remain_above, 
                                from_end=True
                            )
                        )
                        remain_above = 0
                        break
                    else:
                        parts_above.insert(0, prev_chunk['text'])
                        remain_above -= prev_chunk['token_count']
                prev_idx -= 1
            
            # Collect context below
            remain_below = context_size
            parts_below = []
            
            next_idx = i + 1
            while next_idx < len(chunks) and remain_below > 0:
                next_chunk = chunks[next_idx]
                if next_chunk['type'] == 'text':
                    if next_chunk['token_count'] >= remain_below:
                        parts_below.append(
                            self._take_sentences(
                                next_chunk['text'], 
                                remain_below, 
                                from_end=False
                            )
                        )
                        remain_below = 0
                        break
                    else:
                        parts_below.append(next_chunk['text'])
                        remain_below -= next_chunk['token_count']
                next_idx += 1
            
            chunk['context_above'] = ''.join(parts_above)
            chunk['context_below'] = ''.join(parts_below)
        
        return chunks
    
    def _take_sentences(self, text: str, need_tokens: int, from_end: bool = False) -> str:
        """Take sentences from text until reaching token budget"""
        split_pattern = r"([。!?？；！\n]|\. )"
        texts = re.split(split_pattern, text or "", flags=re.DOTALL)
        
        sentences = []
        for i in range(0, len(texts), 2):
            sentence = texts[i]
            if i + 1 < len(texts):
                sentence += texts[i + 1]
            sentences.append(sentence)
        
        collected = ""
        iterator = reversed(sentences) if from_end else sentences
        
        for sentence in iterator:
            test_text = sentence + collected if from_end else collected + sentence
            if self._count_tokens(test_text) >= need_tokens:
                break
            collected = test_text
        
        return collected
    
    def _finalize_chunks(self, internal_chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """Convert internal chunks to final Chunk objects"""
        result = []
        
        for internal in internal_chunks:
            # Combine text with context if present
            full_text = (
                internal.get('context_above', '') +
                internal.get('text', '') +
                internal.get('context_below', '')
            )
            
            if not full_text.strip():
                continue
            
            chunk = Chunk(
                text=full_text,
                chunk_type=internal.get('type', 'text'),
                source_block_ids=[internal['source_block_id']],
                metadata={
                    'token_count': internal.get('token_count', 0),
                    'content_type': internal.get('content_type', 'text'),
                    'context_above': internal.get('context_above', ''),
                    'context_below': internal.get('context_below', '')
                }
            )
            result.append(chunk)
        
        return result

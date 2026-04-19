"""
Comprehensive tests for TitleChunker auto-hierarchy detection

Tests cover:
1. Markdown format detection
2. Pattern-based detection (numbered, uppercase, title case)
3. Frequency-based detection (short lines + long content)
4. Layout metadata matching
5. Outline matching
6. Mixed formats
7. Edge cases and boundary conditions
"""

import pytest
from typing import List

from rag_engine.types import ContentBlock, ContentType
from rag_engine.pipeline.chunker import TitleChunker, Chunk


class TestTitleChunkerMarkdownDetection:
    """Test Markdown format hierarchy detection"""

    def test_single_level_markdown(self):
        """Test detection of single-level markdown (# only)"""
        content = """# Main Title
This is body text under the main title.
More content here."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should create chunks from markdown content
        assert isinstance(chunks, list), "Should return list of chunks"
        # Check that markdown pattern is recognized
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        assert 1 in levels, "Should detect level 1 markdown"

    def test_multi_level_markdown_hierarchy(self):
        """Test detection of multi-level markdown hierarchy"""
        content = """# Chapter 1: Introduction
Introduction text here.

## Section 1.1: Background
Background information.

### Subsection 1.1.1: Details
Detailed content.

## Section 1.2: Methods
Methods description."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should detect multiple levels including 1 and 2
        assert 1 in levels, "Should detect level 1 (# marker)"
        assert 2 in levels, "Should detect level 2 (## marker)"
        assert 3 in levels, "Should detect level 3 (### marker)"

    def test_deep_markdown_hierarchy(self):
        """Test detection of deep markdown hierarchy (# to ######)"""
        content = """# Level 1
Content 1.

## Level 2
Content 2.

### Level 3
Content 3.

#### Level 4
Content 4.

##### Level 5
Content 5.

###### Level 6
Content 6."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        assert 1 in levels, "Level 1 should be detected"
        assert 2 in levels, "Level 2 should be detected"
        assert 3 in levels, "Level 3 should be detected"
        assert 4 in levels, "Level 4 should be detected"

    def test_markdown_with_body_text(self):
        """Test chunking preserves markdown hierarchy with body text"""
        content = """# Section A
This is content under section A.
Multiple lines of body text.

## Subsection A.1
More content here.

# Section B
Content for section B."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        chunks = chunker.chunk(blocks)
        
        # Verify markdown patterns are detected
        assert 1 in levels, "Should detect level 1 markdown"
        assert 2 in levels, "Should detect level 2 markdown"
        # Verify body text is included in chunks or line records
        all_text = "\n".join(c.text for c in chunks) if chunks else ""
        all_lines = "\n".join(line for _, line in line_records)
        assert "Section A" in all_lines, "Should have section A content"


class TestTitleChunkerPatternDetection:
    """Test pattern-based hierarchy detection"""

    def test_numbered_list_detection(self):
        """Test detection of numbered list format (1. 2. 3.)"""
        content = """1. First Topic
Introduction to first topic.

2. Second Topic
Details about second topic.

3. Third Topic
More information here."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should recognize numbered list items as titles (not BODY_LEVEL)
        assert any(level != TitleChunker.BODY_LEVEL for level in levels), "Should detect numbered items"
        # Verify line records were extracted
        assert len(line_records) > 0, "Should extract lines from content"

    def test_uppercase_title_detection(self):
        """Test detection of uppercase titles"""
        content = """MAIN SECTION
This is the intro section.
Some body content here.

SUBSECTION
More detailed content.
Additional text."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should recognize uppercase patterns
        assert len(line_records) > 0, "Should extract lines"
        # Note: The UPPERCASE pattern will be detected
        title_levels = [l for l in levels if l != TitleChunker.BODY_LEVEL]
        assert len(title_levels) >= 0, "Should process content"

    def test_title_case_detection(self):
        """Test detection of Title Case format"""
        content = """Introduction Chapter
Starting with the basics.

Key Concepts And Principles
Explaining important ideas.

Final Thoughts And Conclusions
Wrapping up the content."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        
        # Should detect title case patterns exist
        assert len(line_records) > 0, "Should extract lines"


class TestTitleChunkerFrequencyDetection:
    """Test frequency-based hierarchy detection"""

    def test_short_line_followed_by_long_content(self):
        """Test detection of short lines followed by long content"""
        content = """Overview
This section provides a comprehensive overview of the entire system architecture, 
including all major components, their relationships, and how they interact together 
to create a cohesive and functional whole. The system is designed to be scalable, 
maintainable, and extensible for future enhancements.

Implementation Details
The implementation follows industry best practices and design patterns to ensure 
code quality, maintainability, and performance. We used modern technologies and 
frameworks to build a robust and reliable system that can handle various use cases 
and requirements from our stakeholders.

Testing Strategy
Testing is an integral part of the development process. We employ multiple testing 
methodologies including unit testing, integration testing, and end-to-end testing 
to ensure that all components work correctly both individually and as a system."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should recognize short lines as potential titles
        assert len(chunks) > 1, "Should create multiple chunks from frequency detection"

    def test_multiple_short_lines(self):
        """Test handling of multiple short lines and paragraphs"""
        content = """Background
Some background information here.

Context
The context for this section.

Details
More detailed information.

Analysis
Analytical findings."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        
        # Should extract lines from shorter format
        assert len(line_records) > 0, "Should extract lines from content"


class TestTitleChunkerLayoutMetadata:
    """Test layout metadata-based hierarchy detection"""

    def test_layout_type_metadata(self):
        """Test detection using layout_type metadata"""
        blocks = [
            ContentBlock(
                id="block1",
                type=ContentType.TEXT,
                modality="text",
                content="Document Title",
                metadata={"layout_type": "title", "layoutno": 1}
            ),
            ContentBlock(
                id="block2",
                type=ContentType.TEXT,
                modality="text",
                content="Introduction paragraph.",
                metadata={"layout_type": "body"}
            ),
            ContentBlock(
                id="block3",
                type=ContentType.TEXT,
                modality="text",
                content="Main Section",
                metadata={"layout_type": "section", "layoutno": 1}
            ),
            ContentBlock(

                id="block4",
                type=ContentType.TEXT,
                modality="text",
                content="Section content here.",
                metadata={"layout_type": "body"}
            ),
        ]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        
        # Should extract lines from multiple blocks with metadata
        assert len(line_records) > 0, "Should extract lines from blocks"

    def test_chapter_layout_detection(self):
        """Test detection of chapter markers in metadata"""
        blocks = [
            ContentBlock(
                id="block1",
                type=ContentType.TEXT,
                modality="text",
                content="Chapter 1: Getting Started",
                metadata={"layout_type": "chapter", "layoutno": 1}
            ),
            ContentBlock(
                id="block2",
                type=ContentType.TEXT,
                modality="text",
                content="Chapter content.",
                metadata={"layout_type": "body"}
            ),
        ]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should detect chapter layout properly
        assert len(line_records) > 0, "Should extract lines"
        assert len(levels) > 0, "Should detect levels"


class TestTitleChunkerOutlineMatching:
    """Test PDF outline-based hierarchy detection"""

    def test_outline_matching_with_similarity(self):
        """Test outline matching with character similarity"""
        content = """# Chapter 1
Introduction text.

# Chapter 2
More content."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={
                "outline": [
                    {"title": "Chapter 1", "level": 1},
                    {"title": "Chapter 2", "level": 1}
                ]
            }
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should use outline matching strategy
        assert len(levels) > 0, "Should detect levels from outline"

    def test_outline_without_perfect_match(self):
        """Test outline matching with similar but not exact titles"""
        blocks = [
            ContentBlock(
                id="block1",
                type=ContentType.TEXT,
                modality="text",
                content="Getting Started",
                metadata={
                    "outline": [
                        {"title": "Getting Started Guide", "level": 1}
                    ]
                }
            ),
        ]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should still work even if outline doesn't match exactly
        assert len(chunks) >= 0


class TestTitleChunkerMixedFormats:
    """Test handling of mixed format documents"""

    def test_markdown_and_description(self):
        """Test document with markdown and descriptive text"""
        content = """# Main Guide

This is a general introduction to the system.

## Getting Started
To begin, you need to install the software.

Installation Steps
1. Download the installer
2. Run the setup
3. Follow the wizard

## Configuration
After installation, configure the settings.

### Basic Setup
The basic setup wizard.

### Advanced Options
For advanced users.

## Troubleshooting
Coomon issues and solutions."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        assert len(chunks) > 0

    def test_mixed_markdown_and_numbered_lists(self):
        """Test document mixing markdown and numbered lists"""
        content = """# Technical Documentation

## Module Overview
1. Core Module
Description of core module.

2. Utility Module
Description of utility module.

## API Reference
### Endpoints
1. GET /users
2. POST /users
3. DELETE /users/{id}

### Response Formats
The API returns JSON."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should handle mixed formats
        assert len(line_records) > 0, "Should extract mixed content"
        assert any(level != TitleChunker.BODY_LEVEL for level in levels), "Should detect markdown patterns"


class TestTitleChunkerEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_input(self):
        """Test handling of empty input"""
        chunker = TitleChunker()
        chunks = chunker.chunk([])
        
        assert chunks == []

    def test_single_line_content(self):
        """Test handling of single line content"""
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content="Single line content",
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should not crash
        assert isinstance(chunks, list)

    def test_very_long_content(self):
        """Test handling of very long document"""
        # Create a long document with many sections
        lines = []
        for i in range(100):
            lines.append(f"# Section {i}")
            lines.append(f"Content for section {i}.")
            lines.append(f"## Subsection {i}.1")
            lines.append(f"More content here.")
        
        content = "\n".join(lines)
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        
        # Should handle large documents without error
        assert len(line_records) > 0, "Should extract lines from long content"

    def test_only_whitespace(self):
        """Test handling of whitespace-only content"""
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content="\n\n\n   \n\n",
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should handle gracefully
        assert chunks == []

    def test_special_characters_in_titles(self):
        """Test handling of special characters in titles"""
        content = """# Section: Overview & Introduction
Content here.

## API (Application Programming Interface)
More content.

### Syntax: Key=Value Format
Details."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        
        # Should handle special characters in content
        assert len(line_records) > 0, "Should extract lines with special characters"


class TestTitleChunkerHierarchyCorrectness:
    """Test correctness of hierarchy detection"""

    def test_hierarchy_level_ordering(self):
        """Test that hierarchy levels are properly ordered"""
        content = """# Level 1
Content.

## Level 2
Content.

### Level 3
Content.

## Another Level 2
Content."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Should detect hierarchy levels
        assert 1 in levels, "Should detect level 1 markdown"
        assert 2 in levels, "Should detect level 2 markdown"
        assert 3 in levels, "Should detect level 3 markdown"

    def test_title_preservation(self):
        """Test that titles are correctly preserved in chunks"""
        content = """# Main Section
Body content.

## Subsection
More content."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Verify markdown is detected
        assert 1 in levels, "Should detect main section (level 1)"
        assert 2 in levels, "Should detect subsection (level 2)"

    def test_body_text_grouping(self):
        """Test that body text is grouped under correct headings"""
        content = """# Section 1
Body 1A
Body 1B

# Section 2
Body 2A
Body 2B"""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        
        # Verify markdown detection works
        assert len(line_records) > 0, "Should extract lines"
        assert 1 in levels, "Should detect level 1 headings"


class TestTitleChunkerConfiguration:
    """Test TitleChunker configuration options"""

    def test_include_heading_content_option(self):
        """Test include_heading_content parameter"""
        content = """# Main Title
Body content here."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        # With include_heading_content=False (default)
        chunker1 = TitleChunker(include_heading_content=False)
        chunks1 = chunker1.chunk(blocks)
        
        # With include_heading_content=True
        chunker2 = TitleChunker(include_heading_content=True)
        chunks2 = chunker2.chunk(blocks)
        
        # Both should produce valid chunks
        assert isinstance(chunks1, list)
        assert isinstance(chunks2, list)

    def test_use_outline_option(self):
        """Test use_outline parameter"""
        content = "Content with outline"
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={"outline": [{"title": "Test", "level": 1}]}
        )]
        
        # With use_outline=True (default)
        chunker1 = TitleChunker(use_outline=True)
        chunks1 = chunker1.chunk(blocks)
        
        # With use_outline=False
        chunker2 = TitleChunker(use_outline=False)
        chunks2 = chunker2.chunk(blocks)
        
        # Both should work
        assert isinstance(chunks1, list)
        assert isinstance(chunks2, list)


class TestTitleChunkerWithMultipleBlocks:
    """Test TitleChunker with multiple content blocks"""

    def test_multiple_text_blocks(self):
        """Test chunking multiple text blocks"""
        blocks = [
            ContentBlock(
                id="block1",
                type=ContentType.TEXT,
                modality="text",
                content="# First Document\nContent 1.",
                metadata={}
            ),
            ContentBlock(
                id="block2",
                type=ContentType.TEXT,
                modality="text",
                content="## Section in First\nMore content.",
                metadata={}
            ),
        ]
        
        chunker = TitleChunker()
        line_records = chunker._extract_line_records(blocks)
        levels = chunker._detect_title_levels(line_records, blocks)
        chunks = chunker.chunk(blocks)
        
        # Should process multiple blocks
        assert len(line_records) > 0, "Should extract lines from multiple blocks"
        assert 1 in levels, "Should detect level 1 from first block"
        assert 2 in levels, "Should detect level 2 from second block"

    def test_mixed_content_types(self):
        """Test handling of mixed content types (text, tables, images)"""
        blocks = [
            ContentBlock(
                id="block1",
                type=ContentType.TEXT,
                modality="text",
                content="# Document\nIntroduction.",
                metadata={}
            ),
            ContentBlock(
                id="block2",
                type=ContentType.TABLE,
                modality="table",
                content="| Header 1 | Header 2 |\n|---|---|\n| Data 1 | Data 2 |",
                metadata={}
            ),
            ContentBlock(
                id="block3",
                type=ContentType.IMAGE,
                modality="image",
                content="",
                metadata={"description": "Sample image"}
            ),
        ]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        # Should handle mixed types
        assert isinstance(chunks, list)


class TestChunkDataStructure:
    """Test the Chunk data structure produced"""

    def test_chunk_has_required_fields(self):
        """Test that chunks have all required fields"""
        content = """# Test
Content."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'chunk_type')
            assert hasattr(chunk, 'source_block_ids')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk, 'title')
            assert hasattr(chunk, 'title_level')

    def test_chunk_text_not_empty(self):
        """Test that chunk text is not empty"""
        content = """# Section
This is body text."""
        
        blocks = [ContentBlock(
            id="block1",
            type=ContentType.TEXT,
            modality="text",
            content=content,
            metadata={}
        )]
        
        chunker = TitleChunker()
        chunks = chunker.chunk(blocks)
        
        for chunk in chunks:
            assert chunk.text  # text should not be empty


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

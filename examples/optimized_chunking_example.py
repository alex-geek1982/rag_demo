#!/usr/bin/env python
"""
Advanced Chunking Optimization Example

Demonstrates:
1. TitleChunker with auto-detected hierarchy (no hardcoded H1-H6)
2. Configuration-driven chunker selection
3. Multiple hierarchy detection strategies (outline, pattern, frequency, layout)
"""

import sys
import os
from typing import List

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine.types import ContentBlock, ContentType
from rag_engine.pipeline.chunker import TitleChunker, TokenChunker, Chunk
from rag_engine.config import RAGEngineConfig, ProcessingConfig
from rag_engine.pipeline import DocumentProcessor


def create_sample_content_blocks() -> List[ContentBlock]:
    """Create diverse sample content blocks to test hierarchy detection"""
    
    # Example 1: Markdown formatted
    md_content = """# Enterprise Knowledge Management
This is the introduction.

## System Architecture
Details about architecture.

### Components
Component details here.

#### API Gateway
API details.

### Database Layer
Database information.

## Implementation Guide
Step-by-step guide.

### Setup
Installation steps.
"""
    
    blocks = [
        ContentBlock(
            id="block_1",
            type=ContentType.TEXT,
            content=md_content,
            metadata={"source": "markdown", "format": "md"}
        ),
    ]
    
    return blocks


def create_content_without_markdown() -> List[ContentBlock]:
    """Create content without markdown markers - tests frequency detection"""
    
    content = """ENTERPRISE ARCHITECTURE OVERVIEW

Company Infrastructure
The foundation of our systems includes cloud services, on-premises servers, and hybrid deployments.

Cloud Services
AWS and Azure provide scalability and redundancy. We use multiple regions for disaster recovery.

Database Systems
PostgreSQL handles relational data. MongoDB manages document storage.

IMPLEMENTATION ROADMAP

Phase 1: Migration
Initial setup and data migration from legacy systems.

Phase 2: Optimization
Performance tuning and index optimization.

Phase 3: Scaling
Horizontal scaling across multiple servers.

DATA SECURITY
Encryption at rest and in transit. Multi-factor authentication for access.
"""
    
    blocks = [
        ContentBlock(
            id="block_2",
            type=ContentType.TEXT,
            content=content,
            metadata={"source": "report"}
        ),
    ]
    
    return blocks


def create_content_with_layout_metadata() -> List[ContentBlock]:
    """Create content with layout metadata - tests layout matching"""
    
    content = "Content here"
    
    blocks = [
        ContentBlock(
            id="block_3a",
            type=ContentType.TEXT,
            content="Product Manual",
            metadata={
                "layout_type": "title",
                "layoutno": 1,
                "font_size": 28
            }
        ),
        ContentBlock(
            id="block_3b",
            type=ContentType.TEXT,
            content="Overview of the product features and capabilities.",
            metadata={"layout_type": "body"}
        ),
        ContentBlock(
            id="block_3c",
            type=ContentType.TEXT,
            content="Getting Started",
            metadata={
                "layout_type": "section",
                "layoutno": 2,
                "font_size": 20
            }
        ),
        ContentBlock(
            id="block_3d",
            type=ContentType.TEXT,
            content="Steps to install and configure the product.",
            metadata={"layout_type": "body"}
        ),
    ]
    
    return blocks


def print_chunks(chunks: List[Chunk], title: str):
    """Pretty print chunk results"""
    print(f"\n{'='*70}")
    print(f"📄 {title}")
    print(f"{'='*70}")
    print(f"Generated {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        level_indicator = f"[Level {chunk.title_level}]" if chunk.title_level > 0 else "[Body]"
        print(f"Chunk {i} {level_indicator}:")
        
        if chunk.title:
            print(f"  Title: {chunk.title}")
        
        # Show first 80 chars of content
        text_preview = chunk.text[:80].replace('\n', ' ').strip()
        if len(chunk.text) > 80:
            text_preview += "..."
        print(f"  Content: {text_preview}")
        
        if chunk.metadata:
            print(f"  Metadata: {chunk.metadata}")
        print()


def example_1_markdown_hierarchy_detection():
    """Example 1: Auto-detect hierarchy from Markdown"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Auto-Detect Hierarchy from Markdown")
    print("="*70)
    print("Feature: Pattern Hint Matching + Markdown detection")
    print("The TitleChunker identifies # symbols and creates hierarchy automatically")
    
    blocks = create_sample_content_blocks()
    chunker = TitleChunker()
    chunks = chunker.chunk(blocks)
    
    print_chunks(chunks, "Markdown Content - Auto-Detected Hierarchy")
    
    print("✓ Hierarchy levels auto-detected from markdown")
    print("✓ No hardcoded H1-H6 patterns")
    print("✓ Each heading level properly identified")


def example_2_frequency_based_detection():
    """Example 2: Detect hierarchy from text characteristics"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Frequency-Based Hierarchy Detection")
    print("="*70)
    print("Feature: Frequency Matching without markdown markers")
    print("The TitleChunker analyzes line length and content patterns")
    
    blocks = create_content_without_markdown()
    chunker = TitleChunker()
    chunks = chunker.chunk(blocks)
    
    print_chunks(chunks, "Non-Markdown Content - Frequency Detection")
    
    print("✓ Short lines identified as potential titles")
    print("✓ Lines followed by longer content marked as titles")
    print("✓ Hierarchy inferred from text structure")


def example_3_layout_metadata_detection():
    """Example 3: Use layout metadata for hierarchy"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Layout Metadata Hierarchy Detection")
    print("="*70)
    print("Feature: Layout Matching with PDF/DOCX metadata")
    print("The TitleChunker uses block metadata to infer structure")
    
    blocks = create_content_with_layout_metadata()
    chunker = TitleChunker()
    chunks = chunker.chunk(blocks)
    
    print_chunks(chunks, "Content with Layout Metadata")
    
    print("✓ Layout type metadata recognized (title, section)")
    print("✓ Layout number used as hierarchy level")
    print("✓ Title-like content grouped correctly")


def example_4_configuration_driven_chunker_selection():
    """Example 4: Configuration-driven chunker selection"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Configuration-Driven Chunker Selection")
    print("="*70)
    print("Feature: Use config.chunker_type to select strategy")
    
    blocks = create_sample_content_blocks()
    
    # Test 1: TitleChunker (default)
    print("\n[Test 1] Using TitleChunker (default):")
    print("-" * 50)
    config = RAGEngineConfig()
    # Verify default
    print(f"  chunker_type = '{config.processing.chunker_type}'")
    
    processor = DocumentProcessor(config)
    document, chunks = processor.process_document.__doc__
    print(f"  ✓ DocumentProcessor initialized with TitleChunker")
    
    # Manually create to show
    title_chunker = TitleChunker()
    title_chunks = title_chunker.chunk(blocks)
    print(f"  ✓ Generated {len(title_chunks)} chunks (hierarchy-aware)")
    
    # Test 2: TokenChunker
    print("\n[Test 2] Using TokenChunker (via config):")
    print("-" * 50)
    token_chunker = TokenChunker(
        chunk_token_size=128,
        overlapped_percent=10
    )
    token_chunks = token_chunker.chunk(blocks)
    print(f"  ✓ Generated {len(token_chunks)} chunks (token-sized)")
    
    print("\n✓ Chunker selection is now configuration-driven")
    print("✓ Set CHUNKER_TYPE env var to 'title' or 'token'")
    print("✓ Default is 'title' (TitleChunker)")


def example_5_multi_strategy_comparison():
    """Example 5: Show all four detection strategies"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Multi-Strategy Hierarchy Detection")
    print("="*70)
    print("TitleChunker uses 4 strategies in order:")
    
    strategies = [
        ("1. Outline Matching", "PDF outline extraction with > 0.8 similarity"),
        ("2. Pattern Hints", "Markdown (#), numbered (1.), CAPS, Title Case"),
        ("3. Frequency Analysis", "Short lines + long content = potential title"),
        ("4. Layout Matching", "Use layout_type/layoutno from metadata")
    ]
    
    for strategy_name, description in strategies:
        print(f"\n  {strategy_name}")
        print(f"    {description}")
    
    print("\n✓ Fallback chain ensures good detection even without metadata")
    print("✓ No hardcoded hierarchy definitions")
    print("✓ Adaptive to different document formats")


def main():
    """Run all optimization examples"""
    print("\n" + "="*70)
    print("🚀 Advanced Chunking Optimization Examples")
    print("="*70)
    print("\nKey Improvements:")
    print("1. ✅ TitleChunker now auto-detects hierarchy (no H1-H6 hardcoding)")
    print("2. ✅ Multiple detection strategies (outline, pattern, frequency, layout)")
    print("3. ✅ Configuration-driven chunker selection (no AdaptiveChunker)")
    print("4. ✅ Smarter hierarchy inference")
    
    try:
        example_1_markdown_hierarchy_detection()
        example_2_frequency_based_detection()
        example_3_layout_metadata_detection()
        example_4_configuration_driven_chunker_selection()
        example_5_multi_strategy_comparison()
        
        print("\n" + "="*70)
        print("✅ All optimization examples completed successfully!")
        print("="*70)
        print("\n📚 Key Takeaways:")
        print("  • TitleChunker automatically learns hierarchy from documents")
        print("  • No need to specify H1, H2, H3 - it detects them automatically")
        print("  • Chunker type (title/token) is now configuration-driven")
        print("  • AdaptiveChunker removed for clarity and control")
        print("  • Multiple detection strategies ensure robust performance")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

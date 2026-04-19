"""
Real-world example: Using advanced chunking strategy in RAG pipeline

This example demonstrates:
1. Processing a markdown document with hierarchical structure
2. Using TitleChunker to preserve document hierarchy
3. Integrating chunks into knowledge base builder
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine.config import RAGEngineConfig
from rag_engine.pipeline.document_processor import DocumentProcessor
from rag_engine.pipeline.chunker import AdaptiveChunker, TitleChunker, TokenChunker


def example_1_basic_chunking():
    """Example 1: Basic chunking with auto-detection"""
    print("\n" + "="*70)
    print("Example 1: Adaptive Chunking (Auto-detect Document Structure)")
    print("="*70)
    
    # Initialize config and processor
    config = RAGEngineConfig()
    processor = DocumentProcessor(config)
    
    # Create a sample markdown content
    markdown_content = """# Enterprise Knowledge Management System

## 1. Introduction

The Enterprise Knowledge Management System (EKMS) is designed to help organizations
efficiently capture, organize, and retrieve critical business knowledge.

### 1.1 Key Features
- Centralized knowledge repository
- Advanced search capabilities
- Multi-user collaboration
- Version control

### 1.2 System Architecture
The system consists of three main layers:
- Frontend: Web-based user interface
- Backend: RESTful API services
- Storage: Distributed database

## 2. Implementation Guide

### 2.1 Installation

Step 1: Download the installation package from the official website.
Step 2: Extract the archive to your desired location.
Step 3: Run the setup script: python setup.py install

### 2.2 Configuration

Configure the system by editing the config.yaml file:
- Database connection parameters
- API endpoint settings
- Security configurations

## 3. Usage Examples

### 3.1 Creating Knowledge Articles

Users can create new knowledge articles through the web interface.
Articles support rich text formatting including:
- Bold and italic text
- Lists and tables
- Code snippets
- Embedded multimedia

### 3.2 Searching and Retrieval

The advanced search system supports:
- Full-text search
- Tag-based filtering
- Advanced query syntax
- Relevance scoring

## 4. Conclusion

The Enterprise Knowledge Management System provides a comprehensive solution
for organizations to manage their intellectual capital effectively.
"""
    
    # Save to temporary file
    temp_file = "/tmp/example_doc.md"
    Path(temp_file).parent.mkdir(parents=True, exist_ok=True)
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Process the document
    try:
        document = processor.process_document(
            file_path=temp_file,
            doc_id="ekms_doc",
            doc_title="Enterprise Knowledge Management System"
        )
        
        chunks = document.chunks
        
        print(f"\n✓ Document processed successfully")
        print(f"  - Total chunks: {len(chunks)}")
        
        print(f"\n📑 Chunk Details:")
        for idx, chunk in enumerate(chunks, 1):
            title_display = chunk.title if chunk.title else "(no title)"
            print(f"\n  Chunk {idx}:")
            print(f"    Type: {chunk.chunk_type}")
            print(f"    Context: {title_display}")
            print(f"    Level: {chunk.title_level}")
            print(f"    Text preview: {chunk.text[:60]}...")
            print(f"    Source blocks: {len(chunk.source_block_ids)}")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def example_2_title_chunker_custom():
    """Example 2: Using TitleChunker with custom settings"""
    print("\n" + "="*70)
    print("Example 2: TitleChunker with Custom Hierarchy Settings")
    print("="*70)
    
    from rag_engine.types import Document, ContentBlock, ContentType, ModalityType
    
    # Create document with multi-level headings
    doc = Document(
        id="ml_guide",
        title="Machine Learning Guide",
        source_path="ml_guide.md",
        language="en"
    )
    
    # Add blocks with hierarchy
    content_items = [
        ("# Machine Learning Fundamentals", 'h1'),
        ("Machine learning is a subset of artificial intelligence.", 'text'),
        
        ("## Supervised Learning", 'h2'),
        ("Supervised learning uses labeled data for training.", 'text'),
        
        ("### Regression", 'h3'),
        ("Regression predicts continuous values.", 'text'),
        
        ("### Classification", 'h3'),
        ("Classification predicts discrete categories.", 'text'),
        
        ("## Unsupervised Learning", 'h2'),
        ("Unsupervised learning finds patterns in unlabeled data.", 'text'),
        
        ("### Clustering", 'h3'),
        ("Clustering groups similar data points together.", 'text'),
    ]
    
    for idx, (text, item_type) in enumerate(content_items):
        metadata = {}
        if item_type.startswith('h'):
            level = int(item_type[1])
            metadata = {'block_type': 'heading', 'heading_level': level}
        
        block = ContentBlock(
            id=f"block_{idx}",
            type=ContentType.TEXT,
            content=text,
            modality=ModalityType.TEXT,
            language="en",
            metadata=metadata
        )
        doc.add_content_block(block)
    
    # Create TitleChunker with specific settings
    chunker = TitleChunker(hierarchy_level=3, include_heading_content=False)
    chunks = chunker.chunk(doc.content_blocks)
    
    print(f"\n✓ Created {len(chunks)} chunks with title hierarchy")
    print(f"\n📊 Chunk Structure:")
    
    for idx, chunk in enumerate(chunks, 1):
        indent = "  " * (chunk.title_level - 1) if chunk.title_level > 0 else ""
        print(f"\n{indent}Chunk {idx}:")
        print(f"{indent}  Path: {chunk.title}")
        print(f"{indent}  Text: {chunk.text[:50]}...")


def example_3_token_chunker_custom():
    """Example 3: Using TokenChunker with custom settings"""
    print("\n" + "="*70)
    print("Example 3: TokenChunker with Custom Token Settings")
    print("="*70)
    
    from rag_engine.types import Document, ContentBlock, ContentType, ModalityType
    
    # Create a large flat text document
    doc = Document(
        id="flat_doc",
        title="Technical Documentation",
        source_path="tech_doc.txt",
        language="en"
    )
    
    large_text = """
    API Documentation Overview
    
    This document provides comprehensive guidance on using our REST API for data integration.
    The API supports multiple authentication methods including OAuth 2.0 and API keys.
    
    Base URL
    
    All endpoints are accessed through the base URL: https://api.example.com/v1
    Responses are returned in JSON format with appropriate HTTP status codes.
    
    Authentication
    
    To authenticate, include your API key in the Authorization header as follows:
    Authorization: Bearer YOUR_API_KEY
    
    For OAuth 2.0 authentication, use the token endpoint to obtain access tokens.
    Access tokens expire after 24 hours and can be refreshed using refresh tokens.
    
    Rate Limiting
    
    API requests are rate limited to 1000 requests per hour per API key.
    Rate limit information is included in response headers.
    
    Error Handling
    
    The API returns standard HTTP status codes. 4xx errors indicate client-side issues.
    5xx errors indicate server-side problems. All errors include a descriptive message.
    
    Data Formats
    
    The API accepts and returns data in JSON format. Dates are in ISO 8601 format.
    All string values must be properly escaped for JSON compatibility.
    """ * 2  # Duplicate to create larger content
    
    block = ContentBlock(
        id="large_block",
        type=ContentType.TEXT,
        content=large_text,
        modality=ModalityType.TEXT,
        language="en"
    )
    doc.add_content_block(block)
    
    # Create TokenChunker with aggressive chunking
    chunker = TokenChunker(
        chunk_token_size=256,           # Smaller chunks
        overlapped_percent=20,          # 20% overlap
        delimiters=['\n\n', '\n', '。', '.', '!', '?'],
        table_context_size=0,
        image_context_size=0
    )
    
    chunks = chunker.chunk(doc.content_blocks)
    
    print(f"\n✓ Generated {len(chunks)} token-based chunks")
    print(f"\n📊 Chunk Statistics:")
    
    total_tokens = 0
    for idx, chunk in enumerate(chunks, 1):
        token_count = chunk.metadata.get('token_count', 0)
        total_tokens += token_count
        print(f"\n  Chunk {idx}:")
        print(f"    Tokens: {token_count}")
        print(f"    Text length: {len(chunk.text)} chars")
        print(f"    Preview: {chunk.text[:40]}...")
    
    print(f"\n  Total tokens: {total_tokens}")
    print(f"  Average chunk size: {total_tokens // len(chunks)} tokens")


def example_4_comparing_strategies():
    """Example 4: Side-by-side comparison of both strategies"""
    print("\n" + "="*70)
    print("Example 4: Comparing TitleChunker vs TokenChunker")
    print("="*70)
    
    from rag_engine.types import Document, ContentBlock, ContentType, ModalityType
    
    # Create a structured document
    doc = Document(
        id="comparison_doc",
        title="Comparison Test",
        source_path="test.md",
        language="en"
    )
    
    items = [
        ("# Part A", 'h1'),
        ("Content for part A section 1.", 'text'),
        ("Content for part A section 2.", 'text'),
        ("## Part A.1", 'h2'),
        ("More detailed content under A.1.", 'text'),
        ("# Part B", 'h1'),
        ("Content for part B section 1.", 'text'),
        ("Content for part B section 2.", 'text'),
    ]
    
    for idx, (text, item_type) in enumerate(items):
        metadata = {}
        if item_type.startswith('h'):
            level = int(item_type[1])
            metadata = {'block_type': 'heading', 'heading_level': level}
        
        block = ContentBlock(
            id=f"block_{idx}",
            type=ContentType.TEXT,
            content=text,
            modality=ModalityType.TEXT,
            language="en",
            metadata=metadata
        )
        doc.add_content_block(block)
    
    # Compare both strategies
    title_chunker = TitleChunker(hierarchy_level=2)
    token_chunker = TokenChunker(chunk_token_size=100, overlapped_percent=0)
    
    title_chunks = title_chunker.chunk(doc.content_blocks)
    token_chunks = token_chunker.chunk(doc.content_blocks)
    
    print(f"\n📊 Comparison Results:")
    print(f"\n  TitleChunker:")
    print(f"    - Chunks generated: {len(title_chunks)}")
    print(f"    - Preserves hierarchy: Yes")
    print(f"    - Typical use: Structured documents")
    
    print(f"\n  TokenChunker:")
    print(f"    - Chunks generated: {len(token_chunks)}")
    print(f"    - Preserves hierarchy: No")
    print(f"    - Typical use: Flat text documents")
    
    print(f"\n  TitleChunker output (samples):")
    for i, chunk in enumerate(title_chunks[:3], 1):
        print(f"    {i}. {chunk.title}: {chunk.text[:30]}...")
    
    print(f"\n  TokenChunker output (samples):")
    for i, chunk in enumerate(token_chunks[:3], 1):
        print(f"    {i}. Tokens: {chunk.metadata.get('token_count', 0)}, Text: {chunk.text[:30]}...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG Pipeline: Advanced Chunking Strategy Examples")
    print("="*70)
    
    try:
        example_1_basic_chunking()
        example_2_title_chunker_custom()
        example_3_token_chunker_custom()
        example_4_comparing_strategies()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

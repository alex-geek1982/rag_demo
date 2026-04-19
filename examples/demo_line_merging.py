"""
Demonstration of TitleChunker line merging optimization.

Shows how the improved chunking handles large line spacing in documents,
merging lines that belong to the same sentence.
"""

from rag_engine.types import ContentBlock, ContentType, ModalityType
from rag_engine.pipeline.chunker import TitleChunker


def demo_large_line_spacing_english():
    """Demo: English text with large line spacing"""
    print("=" * 70)
    print("DEMO 1: English Text with Large Line Spacing")
    print("=" * 70)
    
    content = """# Product Overview

This is a long sentence describing the product that
spans multiple lines due to large line spacing in the PDF,
but should be treated as a single coherent phrase.

## Key Features

The system supports multiple content types including
text, images, tables, and mathematical equations,
with intelligent handling of each modality.

## Performance Metrics

Average processing time is 250ms per document with
1000 chunks, showing excellent scalability and performant
delivery of results."""
    
    block = ContentBlock(
        id="demo1",
        type=ContentType.TEXT,
        modality=ModalityType.TEXT,
        content=content,
        metadata={}
    )
    
    chunker = TitleChunker()
    chunks = chunker.chunk([block])
    
    print(f"\nGenerated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Title: {chunk.title or '(no title)'}")
        print(f"  Level: {chunk.title_level}")
        print(f"  Content: {chunk.text[:80]}...")
        print()


def demo_chinese_document_with_spacing():
    """Demo: Chinese document with large line spacing (from user's issue)"""
    print("=" * 70)
    print("DEMO 2: Chinese Document with Large Line Spacing")
    print("=" * 70)
    
    content = """## 项目背景

大宗订单在生产、配送前，没有和用户确认收货时间的沟通机制，导致大宗订单再投率占总订单的9%，
日均246单大宗订单需要二次运输配送，增加运输配送的运营成本。

针对普通客户对大宗订单思意把收的行为，没有举报反馈机制，导致大宗订单的拒收率无法
控制，仓储、运输、配送资源极大浪费。

## 解决方案

建立完善的预约制度和反馈机制，提升用户体验并降低成本。"""
    
    block = ContentBlock(
        id="demo2",
        type=ContentType.TEXT,
        modality=ModalityType.TEXT,
        content=content,
        metadata={}
    )
    
    chunker = TitleChunker()
    chunks = chunker.chunk([block])
    
    print(f"\nGenerated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Title: {chunk.title or '(no title)'}")
        print(f"  Level: {chunk.title_level}")
        print(f"  Content: {chunk.text[:100]}...")
        print()


def demo_line_merging_details():
    """Demo: Show the line merging process in detail"""
    print("=" * 70)
    print("DEMO 3: Line Merging Details")
    print("=" * 70)
    
    content = """Introduction

This is a sentence that
continues on multiple lines
and should be merged into one."""
    
    block = ContentBlock(
        id="demo3",
        type=ContentType.TEXT,
        modality=ModalityType.TEXT,
        content=content,
        metadata={}
    )
    
    chunker = TitleChunker()
    
    # Show raw lines
    raw_lines = content.split('\n')
    print(f"\nRaw lines ({len(raw_lines)} lines):")
    for i, line in enumerate(raw_lines, 1):
        print(f"  {i}: {repr(line)}")
    
    # Show merged lines
    merged_lines = chunker._merge_incomplete_lines(raw_lines)
    print(f"\nMerged lines ({len(merged_lines)} lines):")
    for i, line in enumerate(merged_lines, 1):
        print(f"  {i}: {repr(line)}")
    
    # Show extracted records
    records = chunker._extract_line_records([block])
    print(f"\nExtracted records ({len(records)} records):")
    for i, (_, line) in enumerate(records, 1):
        print(f"  {i}: {repr(line)}")
    
    # Show detected levels
    levels = chunker._detect_title_levels(records, [block])
    print(f"\nDetected levels:")
    for i, (_, line), level in zip(range(len(records)), records, levels):
        level_name = f"Title (level {level})" if level != chunker.BODY_LEVEL else "Body"
        print(f"  {i}: {level_name:20} -> {repr(line[:50])}")
    
    # Show final chunks
    chunks = chunker.chunk([block])
    print(f"\nFinal chunks ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}: {repr(chunk.text[:60])}")


def demo_punctuation_handling():
    """Demo: Punctuation handling in line merging"""
    print("=" * 70)
    print("DEMO 4: Punctuation-based Line Merging")
    print("=" * 70)
    
    chunker = TitleChunker()
    
    test_cases = [
        ["Line without punctuation", "continues here.", "ends."],
        ["Line with period.", "New paragraph."],
        ["Question without mark", "answer here?"],
        ["Chinese text without。", "continues。"],
        ["Already complete.\n", "Next paragraph."],
    ]
    
    for i, lines in enumerate(test_cases, 1):
        merged = chunker._merge_incomplete_lines(lines)
        print(f"\nTest {i}:")
        print(f"  Input:  {lines}")
        print(f"  Output: {merged}")


if __name__ == "__main__":
    demo_large_line_spacing_english()
    print("\n\n")
    
    demo_chinese_document_with_spacing()
    print("\n\n")
    
    demo_line_merging_details()
    print("\n\n")
    
    demo_punctuation_handling()
    
    print("\n" + "=" * 70)
    print("Optimization Benefits:")
    print("=" * 70)
    print("""
1. ✅ Intelligent line merging: Lines with large spacing are correctly merged
2. ✅ Punctuation-aware: Only merges incomplete sentences
3. ✅ Title-aware: Doesn't merge across section boundaries
4. ✅ Multi-language: Works with English and Chinese punctuation
5. ✅ Pattern-based detection: Recognizes markdown, numbered, and CAPS titles
6. ✅ Robust: Handles edge cases with empty lines and irregular spacing
""")

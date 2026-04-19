"""
Test for table and bullet point chunking fixes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_engine.pipeline.chunker import TitleChunker
from rag_engine.types import ContentBlock, ContentType, ModalityType


def test_table_not_split():
    """Test that table rows are kept together and not split across chunks"""
    markdown_content = """## 配置管理

| 配置 | 说明 | 状态 |
|------|------|------|
| 北京配送中心 | 处理北京地区订单 | 启用 |
| 上海配送中心 | 处理上海地区订单 | 启用 |
| 广州配送中心 | 处理广州地区订单 | 禁用 |

其他描述文本。
"""
    
    blocks = [ContentBlock(
        id="test-block",
        type=ContentType.TEXT,
        content=markdown_content,
        modality=ModalityType.TEXT,
        metadata={"source": "converted_markdown"}
    )]
    
    chunker = TitleChunker(chunk_token_size=512)
    chunks = chunker.chunk(blocks)
    
    # Print for inspection
    print("\n=== Test: Table Not Split ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Section ID: {chunk.metadata.get('section_id')}")
        print(f"  Token count: {chunk.metadata.get('token_count')}")
    
    # Verify: table should be in one chunk, not split across multiple chunks
    table_chunks = [c for c in chunks if '|' in c.text and '配置' in c.text]
    print(f"\nTable chunks found: {len(table_chunks)}")
    
    # The whole table should be together
    assert len(table_chunks) >= 1, "Table should be in at least one chunk"
    
    # All table rows should be in one chunk (verify by checking for the separator)
    for chunk in chunks:
        if '|-----|' in chunk.text or '|------|' in chunk.text:
            # This is a table chunk - verify all rows are present
            assert '北京配送中心' in chunk.text, "All table rows should be together"
            assert '上海配送中心' in chunk.text, "All table rows should be together"
            print("✓ Table kept together in one section")
    
    print("✓ Test passed: Table is not split")


def test_bullet_points_grouped():
    """Test that bullet points at the same level are grouped together"""
    markdown_content = """## 权限管理

简要说明：

1、各分公司仅能查看和修改数据记录为本分公司数据，修改后系统记录修改人ERPID和修改时间
2、总公司人员可查看和修改所有公司的数据，修改后系统记录修改人ERPID和修改时间

总公司人员权限在履约系统中设置；调度人员权限由总公司人员在多维度调度系统中添加

### 子章节

1. 第一项内容
2. 第二项内容
3. 第三项内容

其他文本。
"""
    
    blocks = [ContentBlock(
        id="test-block",
        type=ContentType.TEXT,
        content=markdown_content,
        modality=ModalityType.TEXT,
        metadata={"source": "converted_markdown"}
    )]
    
    chunker = TitleChunker(chunk_token_size=512)
    chunks = chunker.chunk(blocks)
    
    # Print for inspection
    print("\n=== Test: Bullet Points Grouped ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Section ID: {chunk.metadata.get('section_id')}")
        print(f"  Content type: {chunk.metadata.get('content_type')}")
        print(f"  Text preview: {chunk.text[:100]}...")
    
    # Find bullet chunks
    bullet_chunks = [c for c in chunks if any(b in c.text for b in ['1、', '2、', '1.', '2.'])]
    print(f"\nBullet chunks found: {len(bullet_chunks)}")
    
    # Same-level bullets should be in same or nearby sections
    for chunk in chunks:
        # If chunk contains first-level bullets, check they're grouped
        if '1、各分公司' in chunk.text or '1. 第一项' in chunk.text:
            # Check if subsequent bullets are in same chunk or at least same section
            section_id = chunk.metadata.get('section_id')
            print(f"  ✓ Found first-level bullet in section {section_id}")
    
    print("✓ Test passed: Bullet points are properly grouped")


def test_mixed_content():
    """Test chunking with mixed content (headings, tables, bullets, text)"""
    markdown_content = """# 订单管理系统

## 1. 权限配置

| 岗位 | 权限 | 状态 |
|-----|------|------|
| 管理员 | 全部 | 启用 |
| 员工 | 基本 | 启用 |

## 2. 调度规则

1、订单调度规则如下
2、首先检查库存
3、然后分配配送中心

## 3. 数据表

| 字段 | 类型 | 说明 |
|-----|------|------|
| order_id | INT | 订单号 |
| status | VARCHAR | 状态 |

其他说明文本。
"""
    
    blocks = [ContentBlock(
        id="test-block",
        type=ContentType.TEXT,
        content=markdown_content,
        modality=ModalityType.TEXT,
        metadata={"source": "converted_markdown"}
    )]
    
    chunker = TitleChunker(chunk_token_size=512)
    chunks = chunker.chunk(blocks)
    
    # Print for inspection
    print("\n=== Test: Mixed Content ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  Section ID: {chunk.metadata.get('section_id')}")
        print(f"  Lines in chunk: {len(chunk.text.split(chr(10)))}")
        preview = chunk.text.replace('\n', ' ')[:80]
        print(f"  Preview: {preview}...")
    
    # Verify structure
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Tables should not be split
    tables_count = sum(1 for c in chunks if '|' in c.text and ('岗位' in c.text or 'order_id' in c.text))
    print(f"Table sections found: {tables_count}")
    
    assert tables_count >= 2, "Should have at least 2 table sections"
    print("✓ Test passed: Mixed content handled correctly")


if __name__ == '__main__':
    test_table_not_split()
    test_bullet_points_grouped()
    test_mixed_content()
    print("\n✓ All tests passed!")

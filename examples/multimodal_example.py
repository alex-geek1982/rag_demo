"""
RAG Engine - Multimodal processing example
"""
from pathlib import Path
from rag_engine.config import RAGEngineConfig
from rag_engine.core import create_engine
from rag_engine.types import ContentType


def multimodal_example():
    """Example showing multimodal content processing"""
    
    # Setup
    config = RAGEngineConfig.from_env()
    engine = create_engine(config)
    
    print("=" * 60)
    print("MULTIMODAL PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create a sample document with mixed content
    sample_dir = Path(config.output_dir) / "multimodal_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a markdown file with various content
    md_content = """# Data Science Report

## Executive Summary
This report presents key findings from our analysis.

## Performance Metrics
| Metric | Value | Unit |
|--------|-------|------|
| Accuracy | 95.2 | % |
| Precision | 0.94 | score |
| Recall | 0.92 | score |
| F1-Score | 0.93 | score |

## Mathematical Analysis
Key formula: P(A|B) = P(B|A) * P(A) / P(B)

This formula represents Bayes' theorem which is fundamental to statistics.

## Key Findings
1. Model performance exceeded expectations
2. Data quality was excellent
3. Training time was optimized
"""
    
    md_path = sample_dir / "report.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    # Process the document
    print("\\n📄 Processing multimodal document...")
    doc = engine.process_document(
        str(md_path),
        doc_title="Data Science Report"
    )
    
    print(f"✅ Processed document with content types:")
    type_count = {}
    for block in doc.content_blocks:
        type_name = block.type.value
        type_count[type_name] = type_count.get(type_name, 0) + 1
    
    for content_type, count in type_count.items():
        print(f"   - {content_type}: {count} blocks")
    
    # Query multimodal content
    print("\\n🔍 Querying multimodal content...")
    
    queries = [
        "What are the performance metrics?",
        "What is the key formula mentioned?",
        "What are the main findings?",
    ]
    
    for query in queries:
        result = engine.query(query, top_k=2)
        print(f"\\nQ: {query}")
        print(f"A: {result.answer[:200]}...")


def image_processing_example():
    """Example showing image processing"""
    
    config = RAGEngineConfig.from_env()
    engine = create_engine(config)
    
    print("\\n" + "=" * 60)
    print("IMAGE PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Note: Requires actual image file
    # For demonstration:
    print("\\nTo process images:")
    print("1. Place image files in data/images/")
    print("2. engine.process_document('path/to/image.jpg')")
    print("3. Images will be processed for VLM analysis")


if __name__ == "__main__":
    try:
        multimodal_example()
        image_processing_example()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

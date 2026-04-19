"""
RAG Engine - Basic usage example
"""
import asyncio
import logging
from pathlib import Path

from rag_engine import RAGEngineConfig, LLMConfig, VisionConfig
from rag_engine.core import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO)


async def main():
    """Main example"""
    
    # 1. Create configuration
    config = RAGEngineConfig.from_env()
    
    # Optionally customize
    config.processing.chunk_size = 1024
    config.processing.enable_multimodal = True
    config.language.default_language = "en"
    
    print("=" * 60)
    print("RAG ENGINE - MULTIMODAL RAG FRAMEWORK")
    print("=" * 60)
    print(f"Configuration: {config.working_dir}")
    print()
    
    # 2. Create RAG engine
    engine = create_engine(config)
    
    # 3. Process documents
    print("📄 Processing documents...")
    
    # Create sample documents for testing
    sample_doc_path = Path(config.output_dir) / "sample.txt"
    sample_doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that focuses on enabling 
    computer systems to learn from data and improve performance without being explicitly programmed.
    
    Key Concepts:
    1. Supervised Learning: Learning from labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through interaction and feedback
    
    Common Algorithms:
    - Linear Regression
    - Decision Trees
    - Neural Networks
    - Support Vector Machines
    
    Applications:
    - Computer Vision
    - Natural Language Processing
    - Recommendation Systems
    - Anomaly Detection
    """
    
    with open(sample_doc_path, 'w') as f:
        f.write(sample_content)
    
    # Process document
    doc = engine.process_document(
        str(sample_doc_path),
        doc_id="sample_doc_001",
        doc_title="ML Fundamentals",
        language="en"
    )
    print(f"✅ Processed document with {len(doc.content_blocks)} content blocks")
    print()
    
    # 4. Query the engine
    print("🔍 Querying the engine...")
    
    queries = [
        "What is machine learning?",
        "What are the main types of learning?",
        "What are common applications of machine learning?"
    ]
    
    for query in queries:
        result = engine.query(query, top_k=3)
        print(f"\n❓ Query: {query}")
        print(f"💡 Answer: {result.answer}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"📚 Sources: {', '.join(result.sources)}")
    
    # 5. Get statistics
    print("\n" + "=" * 60)
    print("📈 Engine Statistics")
    print("=" * 60)
    
    stats = engine.get_statistics()
    print(f"Documents processed: {stats['processed_documents']}")
    print(f"Content blocks: {stats['total_content_blocks']}")
    print(f"Entities in knowledge graph: {stats['knowledge_graph']['entity_count']}")
    print(f"Relationships: {stats['knowledge_graph']['relationship_count']}")
    print()
    
    # 6. Save engine state
    print("💾 Saving engine state...")
    saved_path = engine.save_state()
    print(f"✅ State saved to: {saved_path}")


def main_sync():
    """Synchronous version"""
    # Create configuration
    config = RAGEngineConfig.from_env()
    
    print("=" * 60)
    print("RAG ENGINE - Synchronous Mode")
    print("=" * 60)
    
    # Create engine
    engine = create_engine(config)
    
    # Process document
    sample_doc_path = Path(config.output_dir) / "sample.txt"
    sample_doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_doc_path, 'w') as f:
        f.write("This is a sample document for testing the RAG engine.")
    
    doc = engine.process_document(str(sample_doc_path))
    print(f"Processed: {len(doc.content_blocks)} blocks")
    
    # Query
    result = engine.query("What is this document about?")
    print(f"Query result:\n{result.answer}")


if __name__ == "__main__":
    # Run synchronous example
    try:
        main_sync()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY in .env file")

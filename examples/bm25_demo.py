"""
BM25 Hybrid Retrieval Demo Script

Demonstrates:
1. BM25 full-text search
2. Vector semantic search
3. Weighted fusion of results
4. Different fusion strategies
5. Score normalization effects
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.config import (
    RAGEngineConfig,
    BM25Config,
    HybridRetrievalConfig,
)
from rag_engine.retrieval.bm25_retriever import BM25Retriever, ScoreNormalizer, HybridFuser
from rag_engine.types import RetrievalResult, ContentType


def demo_bm25_basic():
    """Demo 1: Basic BM25 functionality"""
    print("\n" + "="*80)
    print("Demo 1: Basic BM25 Functionality")
    print("="*80)
    
    # Sample documents
    documents = [
        "The order scheduling system prioritizes urgent orders.",
        "Customer orders can be modified before dispatch.",
        "System optimization improves order processing speed.",
        "The scheduling algorithm uses dynamic programming.",
        "Orders are processed in priority order.",
    ]
    
    # Initialize BM25
    config = BM25Config(
        enable_bm25=True,
        k1=1.5,
        b=0.75,
        min_token_length=2,
        language="english"
    )
    
    bm25 = BM25Retriever(config, documents)
    
    # Query
    query = "order scheduling priority"
    results = bm25.search(query, top_k=3)
    
    print(f"\nQuery: {query}\n")
    print("Top-3 Results:")
    for rank, (doc_idx, score) in enumerate(results, 1):
        print(f"  {rank}. [Score: {score:.4f}] {documents[doc_idx]}")


def demo_score_normalization():
    """Demo 2: Score normalization methods"""
    print("\n" + "="*80)
    print("Demo 2: Score Normalization Methods")
    print("="*80)
    
    scores = [0.2, 0.5, 1.5, 3.0, 0.8]
    
    print(f"\nOriginal scores: {scores}\n")
    
    # Min-Max normalization
    normalized_minmax = ScoreNormalizer.normalize_minmax(scores)
    print(f"Min-Max normalized: {[f'{s:.4f}' for s in normalized_minmax]}")
    
    # Sigmoid normalization
    normalized_sigmoid = ScoreNormalizer.normalize_sigmoid(scores, scale=1.0)
    print(f"Sigmoid normalized:  {[f'{s:.4f}' for s in normalized_sigmoid]}")


def demo_fusion_strategies():
    """Demo 3: Different fusion strategies"""
    print("\n" + "="*80)
    print("Demo 3: Fusion Strategies Comparison")
    print("="*80)
    
    # Create mock retrieval results
    vector_results = [
        RetrievalResult(doc_id="doc1", score=0.9, content="Content 1", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc2", score=0.7, content="Content 2", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc3", score=0.5, content="Content 3", content_type=ContentType.TEXT),
    ]
    
    bm25_results = [
        RetrievalResult(doc_id="doc1", score=0.8, content="Content 1", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc4", score=0.75, content="Content 4", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc2", score=0.6, content="Content 2", content_type=ContentType.TEXT),
    ]
    
    print("\nVector Results:")
    for i, r in enumerate(vector_results, 1):
        print(f"  {i}. {r.doc_id}: {r.score:.4f}")
    
    print("\nBM25 Results:")
    for i, r in enumerate(bm25_results, 1):
        print(f"  {i}. {r.doc_id}: {r.score:.4f}")
    
    # Test each strategy
    strategies = ['weighted_avg', 'rrf', 'max', 'min']
    
    for strategy in strategies:
        print(f"\n--- Fusion Strategy: {strategy} ---")
        
        config = HybridRetrievalConfig(
            fusion_strategy=strategy,
            vector_weight=0.5,
            bm25_weight=0.5,
            normalization_method="minmax",
            rrf_k=60.0,
        )
        
        fuser = HybridFuser(config)
        fused = fuser.fuse(vector_results, bm25_results, top_k=3)
        
        print(f"Fused Results ({strategy}):")
        for i, r in enumerate(fused, 1):
            print(f"  {i}. {r.doc_id}: {r.score:.4f} [{r.metadata.get('retrieval_channel', 'unknown')}]")


def demo_weight_effects():
    """Demo 4: Effect of different weight combinations"""
    print("\n" + "="*80)
    print("Demo 4: Weight Configuration Effects")
    print("="*80)
    
    # Mock results
    vector_results = [
        RetrievalResult(doc_id="doc1", score=0.95, content="Semantic match", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc2", score=0.70, content="Related", content_type=ContentType.TEXT),
    ]
    
    bm25_results = [
        RetrievalResult(doc_id="doc3", score=1.0, content="Exact keyword match", content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc1", score=0.80, content="Semantic match", content_type=ContentType.TEXT),
    ]
    
    # Test different weights
    weight_configs = [
        (0.5, 0.5, "Balanced"),
        (0.7, 0.3, "Vector-heavy (semantic focus)"),
        (0.3, 0.7, "BM25-heavy (keyword focus)"),
        (0.9, 0.1, "Almost vector-only"),
    ]
    
    for v_weight, b_weight, description in weight_configs:
        print(f"\n--- {description} (Vector: {v_weight}, BM25: {b_weight}) ---")
        
        config = HybridRetrievalConfig(
            fusion_strategy="weighted_avg",
            vector_weight=v_weight,
            bm25_weight=b_weight,
        )
        
        fuser = HybridFuser(config)
        fused = fuser.fuse(vector_results, bm25_results, top_k=3)
        
        for i, r in enumerate(fused, 1):
            print(f"  {i}. {r.doc_id}: {r.score:.4f}")


def demo_config_management():
    """Demo 5: Configuration management"""
    print("\n" + "="*80)
    print("Demo 5: Configuration Management")
    print("="*80)
    
    config = RAGEngineConfig()
    
    print(f"\nDefault Configuration:")
    print(f"  BM25 Enabled: {config.bm25.enable_bm25}")
    print(f"  BM25 k1: {config.bm25.k1}")
    print(f"  BM25 b: {config.bm25.b}")
    print(f"  Fusion Strategy: {config.hybrid_retrieval.fusion_strategy}")
    print(f"  Vector Weight: {config.hybrid_retrieval.vector_weight}")
    print(f"  BM25 Weight: {config.hybrid_retrieval.bm25_weight}")
    print(f"  Normalization Method: {config.hybrid_retrieval.normalization_method}")
    
    # Modify configuration
    print(f"\n\nModifying Configuration...")
    config.hybrid_retrieval.vector_weight = 0.7
    config.hybrid_retrieval.bm25_weight = 0.3
    config.hybrid_retrieval.fusion_strategy = "rrf"
    
    print(f"\nUpdated Configuration:")
    print(f"  Fusion Strategy: {config.hybrid_retrieval.fusion_strategy}")
    print(f"  Vector Weight: {config.hybrid_retrieval.vector_weight}")
    print(f"  BM25 Weight: {config.hybrid_retrieval.bm25_weight}")


def demo_hybrid_scenario():
    """Demo 6: Real-world hybrid retrieval scenario"""
    print("\n" + "="*80)
    print("Demo 6: Real-World Hybrid Scenario")
    print("="*80)
    
    # Simulate document set from an order management system
    documents = [
        "Order #12345 was scheduled for delivery on Monday at 2PM.",
        "The scheduling algorithm optimizes delivery routes using machine learning.",
        "Customers can modify their order status through the web portal.",
        "Orders in the queue are processed by priority level.",
        "The system automatically reschedules orders if delivery is delayed.",
        "Optimization techniques improve the scheduling efficiency by 15%.",
        "Order cancellation is allowed up to 24 hours before scheduled delivery.",
    ]
    
    bm25_config = BM25Config(enable_bm25=True)
    bm25 = BM25Retriever(bm25_config, documents)
    
    # Test query
    query = "order scheduling optimization"
    
    print(f"\nQuery: '{query}'\n")
    
    # BM25 results
    bm25_results_raw = bm25.search(query, top_k=5)
    bm25_results = [
        RetrievalResult(
            doc_id=f"doc{idx}",
            score=min(1.0, score / (score + 1.0)),  # Normalize to [0, 1]
            content=documents[idx],
            content_type=ContentType.TEXT,
            metadata={"retrieval_channel": "bm25"}
        )
        for idx, score in bm25_results_raw
    ]
    
    # Simulate vector results (would come from embedding search in real scenario)
    vector_results = [
        RetrievalResult(doc_id="doc1", score=0.75, content=documents[0], content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc2", score=0.88, content=documents[1], content_type=ContentType.TEXT),
        RetrievalResult(doc_id="doc4", score=0.65, content=documents[3], content_type=ContentType.TEXT),
    ]
    
    print("Vector Search Results (simulated):")
    for r in vector_results:
        print(f"  - {r.doc_id}: {r.score:.4f}")
    
    print("\nBM25 Search Results:")
    for r in bm25_results[:3]:
        print(f"  - {r.doc_id}: {r.score:.4f}")
    
    # Fuse with weighted average
    config = HybridRetrievalConfig(
        fusion_strategy="weighted_avg",
        vector_weight=0.5,
        bm25_weight=0.5,
        normalization_method="minmax",
    )
    
    fuser = HybridFuser(config)
    fused = fuser.fuse(vector_results, bm25_results, top_k=4)
    
    print("\nFused Results (50% vector + 50% BM25):")
    for i, r in enumerate(fused, 1):
        print(f"  {i}. {r.doc_id} [Score: {r.score:.4f}]")
        print(f"     {r.content[:60]}...")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("BM25 Full-Text Search & Hybrid Fusion Demonstration")
    print("="*80)
    
    try:
        demo_bm25_basic()
        demo_score_normalization()
        demo_fusion_strategies()
        demo_weight_effects()
        demo_config_management()
        demo_hybrid_scenario()
        
        print("\n" + "="*80)
        print("All demos completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

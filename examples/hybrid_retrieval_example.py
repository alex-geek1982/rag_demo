"""
BM25 + Vector Hybrid Retrieval Integration Example

This example shows how to use BM25 and vector search together in a real RAG pipeline.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List

from rag_engine.config import RAGEngineConfig, HybridRetrievalConfig
from rag_engine.types import RetrievalResult, ContentType
from rag_engine.retrieval.bm25_retriever import HybridFuser, ScoreNormalizer


class RAGQASystem:
    """Simple RAG QA System using hybrid retrieval"""
    
    def __init__(self, config: RAGEngineConfig):
        self.config = config
    
    def create_sample_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create sample knowledge base documents"""
        documents = [
            {
                "id": "doc1",
                "title": "Order Management System Overview",
                "content": "The order management system handles order creation, modification, and fulfillment. "
                          "Orders can be modified before they enter the scheduling queue.",
            },
            {
                "id": "doc2", 
                "title": "Scheduling Algorithm",
                "content": "The scheduling algorithm optimizes delivery routes using dynamic programming. "
                          "It considers traffic patterns, distance, and delivery time windows.",
            },
            {
                "id": "doc3",
                "title": "Order Modification Policy",
                "content": "Orders can be modified up to 24 hours before scheduled delivery. "
                          "Modifications include address changes and time window adjustments.",
            },
            {
                "id": "doc4",
                "title": "Priority Queue Management",
                "content": "Orders are processed in priority order based on urgency and delivery deadline. "
                          "Express orders get priority in the scheduling queue.",
            },
            {
                "id": "doc5",
                "title": "System Performance Optimization",
                "content": "Optimization techniques improve scheduling efficiency by 15% on average. "
                          "The system uses machine learning for route prediction.",
            },
        ]
        return documents
    
    def simulate_vector_search(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """Simulate vector search results (in real system, would use embedding model)"""
        # Simple keyword matching as mock for vector search
        query_words = set(query.lower().split())
        scores = {}
        
        for doc in documents:
            content_words = set((doc["title"] + " " + doc["content"]).lower().split())
            overlap = len(query_words & content_words) / len(query_words) if query_words else 0
            scores[doc["id"]] = overlap * 0.7 + 0.2  # Add base score
        
        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_results:
            doc = next(d for d in documents if d["id"] == doc_id)
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=min(0.99, score),
                    content=doc["content"],
                    content_type=ContentType.TEXT,
                    metadata={"title": doc["title"], "retrieval_channel": "vector"}
                )
            )
        
        return results
    
    def simulate_bm25_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """Simulate BM25 search results (keyword matching)"""
        from rag_engine.retrieval.bm25_retriever import BM25Retriever
        from rag_engine.config import BM25Config
        
        # Index documents
        doc_contents = [d["content"] for d in documents]
        config = BM25Config(enable_bm25=True)
        
        try:
            bm25 = BM25Retriever(config, doc_contents)
            results_raw = bm25.search(query, top_k=top_k)
            
            results = []
            for doc_idx, score in results_raw:
                # Normalize BM25 score
                normalized_score = min(1.0, score / (score + 1.0))
                doc = documents[doc_idx]
                results.append(
                    RetrievalResult(
                        doc_id=doc["id"],
                        score=normalized_score,
                        content=doc["content"],
                        content_type=ContentType.TEXT,
                        metadata={"title": doc["title"], "retrieval_channel": "bm25"}
                    )
                )
            
            return results
        except Exception as e:
            print(f"BM25 search failed: {e}")
            return []
    
    def retrieve_and_fuse(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        fusion_strategy: str = "weighted_avg",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """Perform vector + BM25 retrieval and fusion"""
        
        print(f"Query: '{query}'\n")
        
        # Step 1: Vector search
        print("1. Performing Vector Search...")
        vector_results = self.simulate_vector_search(query, documents, top_k=3)
        for r in vector_results:
            print(f"   - {r.doc_id}: {r.score:.4f} ('{r.metadata.get('title', '')}')")
        
        # Step 2: BM25 search
        print("\n2. Performing BM25 Full-Text Search...")
        bm25_results = self.simulate_bm25_search(query, documents, top_k=3)
        for r in bm25_results:
            print(f"   - {r.doc_id}: {r.score:.4f} ('{r.metadata.get('title', '')}')")
        
        # Step 3: Fuse results
        print(f"\n3. Fusing Results ({fusion_strategy}, V:{vector_weight:.1f}/B:{bm25_weight:.1f})...")
        
        config = HybridRetrievalConfig(
            fusion_strategy=fusion_strategy,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            normalization_method="minmax",
        )
        
        fuser = HybridFuser(config)
        fused_results = fuser.fuse(vector_results, bm25_results, top_k=top_k)
        
        return fused_results
    
    def print_results(self, results: List[RetrievalResult]):
        """Print fusion results"""
        print("\n📊 Final Fused Results:")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.doc_id} (Score: {result.score:.4f})")
            print(f"   Sources: {result.metadata.get('retrieval_channel', 'unknown')}")
            print(f"   Title: {result.metadata.get('title', 'N/A')}")
            print(f"   Content: {result.content[:100]}...")


def demo_order_management_system():
    """Demo: Order Management System RAG"""
    print("\n" + "="*80)
    print("Order Management System - Hybrid Retrieval Demo")
    print("="*80)
    
    config = RAGEngineConfig()
    system = RAGQASystem(config)
    
    # Create knowledge base
    documents = system.create_sample_knowledge_base()
    print(f"\n📚 Knowledge Base: {len(documents)} documents indexed")
    
    # Test queries
    test_queries = [
        {
            "query": "Can I modify an order before delivery?",
            "description": "Mixed: Semantic + Keyword",
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
        },
        {
            "query": "order scheduling optimization",
            "description": "Keyword-focused query",
            "vector_weight": 0.3,
            "bm25_weight": 0.7,
        },
        {
            "query": "What are the system improvements?",
            "description": "Semantic-focused query",
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
        },
    ]
    
    for query_info in test_queries:
        print(f"\n\n{'='*80}")
        print(f"Query ({query_info['description']})")
        print(f"{'='*80}")
        
        results = system.retrieve_and_fuse(
            query=query_info["query"],
            documents=documents,
            fusion_strategy="weighted_avg",
            vector_weight=query_info["vector_weight"],
            bm25_weight=query_info["bm25_weight"],
            top_k=3
        )
        
        system.print_results(results)


def demo_fusion_strategies_comparison():
    """Demo: Compare different fusion strategies"""
    print("\n" + "="*80)
    print("Fusion Strategies Comparison")
    print("="*80)
    
    config = RAGEngineConfig()
    system = RAGQASystem(config)
    documents = system.create_sample_knowledge_base()
    
    query = "scheduling and optimization"
    strategies = ["weighted_avg", "rrf", "max"]
    
    for strategy in strategies:
        print(f"\n\n{'-'*80}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'-'*80}")
        
        results = system.retrieve_and_fuse(
            query=query,
            documents=documents,
            fusion_strategy=strategy,
            vector_weight=0.5,
            bm25_weight=0.5,
            top_k=3
        )
        
        system.print_results(results)


def main():
    """Run all demos"""
    demo_order_management_system()
    demo_fusion_strategies_comparison()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

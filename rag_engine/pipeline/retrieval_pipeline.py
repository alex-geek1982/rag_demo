"""
Retrieval Pipeline - Unified retrieval, reranking, and answer generation
"""

import logging
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from rag_engine.core.prompts import ENTITY_EXTRACTION_USER_PROMPT

from ..config import RAGEngineConfig
from ..retrieval import OpenAIEmbedding
from ..retrieval.bm25_retriever import HybridFuser
from ..types import RetrievalResult, ContentBlock, Entity, ContentType
from ..storage import ChromaKnowledgeBase, KuzuGraphStore

logger = logging.getLogger(__name__)


class LocalReranker:
    """Local Ollama-based reranker for scoring and ranking results."""

    def __init__(self, base_url: str, model: str):
        """
        Initialize local reranker.

        Args:
            base_url: Ollama base URL
            model: Reranker model name
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def rerank(
        self, query: str, docs: List[RetrievalResult], top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank documents using local Ollama reranker.

        Falls back to embedding-based scoring if reranker API fails.

        Args:
            query: Query string
            docs: List of retrieval results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not docs:
            return []

        # Try reranker API first
        payload = {
            "model": self.model,
            "query": query,
            "top_n": top_k,
            "documents": [doc.content[:2000] for doc in docs],
        }

        try:
            response = self.session.post(f"{self.base_url}/api/rerank", json=payload, timeout=90)
            if response.ok:
                data = response.json()
                for item in data.get("results", []):
                    index = int(item.get("index", 0))
                    score = float(item.get("relevance_score", item.get("score", 0.0)))
                    docs[index].score = score
                    docs[index].metadata["retrieval_channel"] = (
                        docs[index].metadata.get("retrieval_channel", "hybrid") + "+ollama-rerank"
                    )
                return sorted(docs, key=lambda item: item.score, reverse=True)[:top_k]
        except Exception:
            logger.debug("Reranker API failed, using embedding fallback")
            pass

        # Fallback: use embedding-based scoring
        embed_payload = {
            "model": self.model,
            "input": [f"query: {query}"] + [f"passage: {doc.content[:2000]}" for doc in docs],
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/embed", json=embed_payload, timeout=90
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            query_vector = embeddings[0]

            for doc, doc_vector in zip(docs, embeddings[1:]):
                rerank_score = self._cosine_similarity(query_vector, doc_vector)
                doc.score = (doc.score * 0.35) + (rerank_score * 0.65)
                doc.metadata["reranked_by"] = self.model

            reranked = sorted(docs, key=lambda item: item.score, reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Both reranker methods failed: {e}")
            return sorted(docs, key=lambda item: item.score, reverse=True)[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.asarray(vec1, dtype=float)
        b = np.asarray(vec2, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


class LocalAnswerGenerator:
    """Local LLM-based answer generator."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ):
        """
        Initialize answer generator.

        Args:
            base_url: LLM API base URL
            api_key: API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
        """
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            logger.error("openai library not installed")
            raise
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self, query: str, vector_result: List[RetrievalResult], graph_result: List[RetrievalResult]
    ) -> str:
        """
        Generate answer from query and retrieved documents.

        Args:
            query: Query string
            vector_result: Vector search results (text chunks)
            graph_result: Graph search results (relationships/entities)

        Returns:
            Generated answer
        """
        if not vector_result and not graph_result:
            return "No relevant information found in the knowledge base."

        # Separate text chunks from relationship data
        text_chunks = [doc for doc in vector_result if doc.content_type == ContentType.TEXT]
        relationships = [doc for doc in graph_result if "relationship" in doc.metadata.get("retrieval_channel", "") or "graph" in doc.metadata.get("retrieval_channel", "")]
        entity_chunks = [doc for doc in graph_result if "entity" in doc.metadata.get("retrieval_channel", "")]

        # Build text chunks context
        text_parts = []
        for idx, doc in enumerate(text_chunks, 1):
            channel = doc.metadata.get("retrieval_channel", "vector")
            preview = doc.content[:1000].replace("\n", " ")
            text_parts.append(
                f"[Text-{idx}] (score={doc.score:.4f}, source={channel})\n{preview}"
            )

        # Build relationship context
        rel_parts = []
        for idx, doc in enumerate(relationships, 1):
            rel_type = doc.metadata.get("relationship_type", "related")
            strength = doc.metadata.get("strength", 0.0)
            rel_parts.append(
                f"[Rel-{idx}] {doc.content} (relevance={doc.score:.4f}, strength={strength:.2f})"
            )

        # Build entity context
        entity_parts = []
        for idx, doc in enumerate(entity_chunks, 1):
            entity_type = doc.metadata.get("entity_type", "unknown")
            pagerank = doc.metadata.get("pagerank", 0.0)
            entity_parts.append(
                f"[Entity-{idx}] {doc.doc_id} (type={entity_type}, relevance={doc.score:.4f}, pagerank={pagerank:.3f})\n{doc.content[:500]}"
            )

        # Construct the context sections
        context_sections = []

        if text_parts:
            context_sections.append(
                "=== TEXT CHUNKS ===\n" + "\n---\n".join(text_parts[:8])  # Limit to top 8 text chunks
            )

        if entity_parts:
            context_sections.append(
                "=== ENTITIES ===\n" + "\n".join(entity_parts[:5])  # Limit to top 5 entities
            )

        if rel_parts:
            context_sections.append(
                "=== RELATIONSHIPS ===\n" + "\n".join(rel_parts[:5])  # Limit to top 5 relationships
            )

        full_context = "\n\n".join(context_sections)

        prompt = f"""You are an enterprise RAG assistant. Answer the question based ONLY on the provided context.

Instructions:
- Use only information from the context below
- If the context doesn't contain enough information, state "The provided documents do not contain sufficient information to answer this question."
- Cite sources using the reference tags (e.g., Text-1, Entity-2, Rel-3)
- Be concise but comprehensive

Question: {query}

Context:
{full_context}

Answer:""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise RAG问答助手. Answer based ONLY on the retrieved context. Cite sources in your response.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Answer generation failed: {str(e)}"


class RetrievalPipeline:
    """
    Unified retrieval and answer generation pipeline.

    Responsibility: Orchestrate vector search, graph search, reranking, and answer generation.

    This class brings together:
    - Vector similarity search (Chroma)
    - Graph-based search (Kuzu)
    - Result merging and deduplication
    - Reranking
    - Answer generation
    """

    def __init__(self, config: RAGEngineConfig):
        """
        Initialize retrieval pipeline.

        Args:
            config: RAG engine configuration
        """
        self.config = config
        self.embedding_provider = None
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Initialize embedding provider."""
        if not self.config.embedding.api_key:
            raise ValueError("OpenAI API key not configured")

        self.embedding_provider = OpenAIEmbedding(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            base_url=self.config.embedding.base_url,
        )

    async def retrieve_hybrid(
        self,
        query: str,
        chroma_kb: ChromaKnowledgeBase,
        kuzu_store: KuzuGraphStore,
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        Perform hybrid retrieval combining vector, BM25, and graph search.
        
        Workflow:
        1. Vector search for semantic similarity
        2. BM25 search for exact/keyword matching
        3. Entity and graph search for relationship data
        4. Fuse results using configurable weights

        Args:
            query: Query string
            chroma_kb: Chroma knowledge base
            kuzu_store: Kuzu graph store
            top_k: Number of results to return

        Returns:
            Tuple of (fused_results, graph_results)
            - fused_results: Merged vector + BM25 results using weighted fusion
            - graph_results: Graph-based relationship data
        """
        try:
            # ========== Vector Search ==========
            logger.info(f"[Hybrid Retrieval] Running vector search for query: {query[:50]}...")
            query_vector = self.embedding_provider.embed_text_sync([query])[0].tolist()
            vector_results = chroma_kb.search(query_vector, top_k=max(top_k * 2, 6))
            logger.debug(f"  Vector search returned {len(vector_results)} results")
            
            # ========== BM25 Search ==========
            bm25_results = []
            if self.config.bm25.enable_bm25 and chroma_kb.bm25_retriever is not None:
                logger.info(f"[Hybrid Retrieval] Running BM25 search...")
                bm25_results = chroma_kb.search_bm25(query, top_k=max(top_k * 2, 6))
                logger.debug(f"  BM25 search returned {len(bm25_results)} results")
            else:
                logger.debug("BM25 search disabled or not initialized")
            
            # ========== Weighted Fusion ==========
            logger.info(f"[Hybrid Retrieval] Fusing vector and BM25 results...")
            fused_results = self._fuse_results(vector_results, bm25_results, top_k)
            logger.info(f"  Fusion complete: {len(fused_results)} results after deduplication and fusion")
            
            # ========== Entity Search ==========
            entity_results = chroma_kb.search_entities(query_vector, top_k=max(top_k, 3))
            logger.debug(f"  Entity search returned {len(entity_results)} results")

            # ========== Graph Search ==========
            graph_search_result = await kuzu_store.search(entity_results, top_k=max(2, top_k))
            graph_results = self._convert_graph_results_to_retrieval_results(graph_search_result)
            logger.debug(f"  Graph search returned {len(graph_results)} results")

            return fused_results, graph_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
            return [], []

    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Fuse vector and BM25 results using HybridFuser.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of final results to return
            
        Returns:
            Fused results
        """
        try:
            fuser = HybridFuser(self.config.hybrid_retrieval)
            fused = fuser.fuse(
                vector_results=vector_results,
                bm25_results=bm25_results,
                graph_results=[],  # Graph results handled separately
                top_k=top_k
            )
            return fused
        except Exception as e:
            logger.error(f"Result fusion failed: {e}", exc_info=True)
            # Fallback: combine and sort by score
            combined = vector_results + bm25_results
            return sorted(combined, key=lambda x: x.score, reverse=True)[:top_k]

    @staticmethod
    def _convert_graph_results_to_retrieval_results(
        graph_result: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """
        Convert structured graph search results to RetrievalResult objects.

        Args:
            graph_result: Dict from kuzu_store.search() with entities and relationships

        Returns:
            List of RetrievalResult objects for compatibility with merge pipeline
        """
        results = []

        # Convert entities to RetrievalResult
        for entity in graph_result.get("entities", []):
            description = entity.get("Description", "")
            score = float(entity.get("Score", 0.5))

            result = RetrievalResult(
                doc_id=entity.get("Entity", "unknown"),
                score=score,
                content=description,
                content_type=ContentType.TEXT,
                metadata={
                    "retrieval_channel": "graph/kuzu/entity",
                    "entity_type": entity.get("Type", "unknown"),
                    "pagerank": entity.get("PageRank", 0.0),
                },
            )
            results.append(result)

        # Convert relationships to RetrievalResult
        for rel in graph_result.get("relationships", []):
            from_entity = rel.get("From Entity", "")
            to_entity = rel.get("To Entity", "")
            rel_type = rel.get("Type", "related")
            content = f"[Relationship] {from_entity} --[{rel_type}]--> {to_entity}"
            score = float(rel.get("Score", 0.3))

            result = RetrievalResult(
                doc_id=f"{from_entity}_{rel_type}_{to_entity}",
                score=score,
                content=content,
                content_type=ContentType.TEXT,
                metadata={
                    "retrieval_channel": "graph/kuzu/relationship",
                    "relationship_type": rel_type,
                    "strength": rel.get("Strength", 0.0),
                },
            )
            results.append(result)

        return results

    @staticmethod
    def _merge_results(*groups: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Merge results from multiple sources, deduplicating and combining scores.

        Args:
            *groups: Multiple lists of retrieval results

        Returns:
            Merged list with deduplication
        """
        merged: Dict[str, RetrievalResult] = {}

        for group in groups:
            for item in group:
                item_channels = set(
                    str(item.metadata.get("retrieval_channel", "hybrid")).split("+")
                )
                existing = merged.get(item.doc_id)

                if existing is None:
                    item.metadata["retrieval_channel"] = "+".join(sorted(item_channels))
                    merged[item.doc_id] = item
                    continue

                existing_channels = set(
                    str(existing.metadata.get("retrieval_channel", "hybrid")).split("+")
                )
                merged_channels = "+".join(sorted(existing_channels | item_channels))

                if item.score > existing.score:
                    item.metadata.update(existing.metadata)
                    item.metadata["retrieval_channel"] = merged_channels
                    merged[item.doc_id] = item
                else:
                    existing.metadata.update(item.metadata)
                    existing.metadata["retrieval_channel"] = merged_channels
                    existing.score = max(existing.score, item.score)

        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def extract_entities(self, query: str, generator: LocalAnswerGenerator) -> List[str]:
        """
        Extract key entity names from the query using LLM.

        Args:
            query: Query string
            generator: Answer generator with LLM client

        Returns:
            List of extracted entity names
        """
        prompt = """
从以下查询中提取关键实体名称。返回 JSON 格式，键为 "entities"，值为字符串列表。

查询：{query}

只返回 JSON，不要其他内容。
""".strip()

        try:
            response = generator.client.chat.completions.create(
                model=generator.model,
                temperature=0.1,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_USER_PROMPT},
                    {"role": "user", "content": prompt.format(query=query)},
                ],
            )
            result = response.choices[0].message.content
            import json

            data = json.loads(result.strip())
            entities = data.get("entities", [])
            return entities if isinstance(entities, list) else []
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    async def run_query(
        self,
        query: str,
        chroma_kb: ChromaKnowledgeBase,
        kuzu_store: KuzuGraphStore,
        reranker: Optional[Any],
        generator: LocalAnswerGenerator,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute full query pipeline: entity extraction -> retrieval -> optional reranking -> answer generation.

        Args:
            query: Query string
            chroma_kb: Chroma knowledge base
            kuzu_store: Kuzu graph store
            entities: Entities for graph search (deprecated, now extracted from query)
            reranker: Optional reranker instance
            generator: Answer generator instance
            top_k: Number of final results

        Returns:
            Query result dict with answer and top docs
        """
        logger.info(f"Processing query: {query}")

        vector_result, graph_result = await self.retrieve_hybrid(
            query, chroma_kb, kuzu_store, top_k
        )

        reranked = (
            reranker.rerank(query, vector_result, top_k) if reranker else vector_result[:top_k]
        )

        answer = generator.generate(query, vector_result, graph_result)

        return {
            "query": query,
            "answer": answer,
            "top_docs": [
                {
                    "doc_id": doc.doc_id,
                    "score": round(float(doc.score), 4),
                    "channel": doc.metadata.get("retrieval_channel", "hybrid"),
                    "page_num": doc.metadata.get("page_num"),
                    "preview": doc.content[:240].replace("\n", " "),
                }
                for doc in reranked
            ],
        }

"""
Chroma Vector Knowledge Base - Isolated vector database operations
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pickle
import json

from ..types import RetrievalResult, ContentType

logger = logging.getLogger(__name__)

COLLECTION_NAME = "jd_order_prd"
ENTITIES_COLLECTION_NAME = "entities"


class ChromaKnowledgeBase:
    """
    Chroma-based vector knowledge base for semantic search.
    
    Responsibilities:
    - Persist and manage vector embeddings
    - Provide semantic search capabilities
    - Support BM25 full-text search
    - Independent rebuild from content blocks
    
    This class is completely decoupled from RAGEngine and other components.
    """

    def __init__(self, db_path: Path):
        """
        Initialize Chroma knowledge base.
        
        Args:
            db_path: Path to persistent Chroma database
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = None
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
        
        # BM25 support
        self.bm25_retriever = None
        self.bm25_doc_ids: List[str] = []
        self._init_bm25_from_disk()

    def get_or_create_collection(self, name: str = COLLECTION_NAME) -> Any:
        """
        Get or create a Chroma collection.
        
        Args:
            name: Collection name
            
        Returns:
            Chroma collection
        """
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def _init_bm25_from_disk(self) -> None:
        """Load BM25 retriever and id mapping from disk if available."""
        try:
            bm25_path = self.db_path / "bm25_index.pkl"
            if bm25_path.exists():
                with open(bm25_path, 'rb') as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    self.bm25_retriever = payload.get('bm25_retriever')
                    self.bm25_doc_ids = payload.get('bm25_doc_ids', [])
                else:
                    self.bm25_retriever = payload
                logger.info("BM25 index loaded from disk")
            else:
                logger.debug("No BM25 index found on disk")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            self.bm25_retriever = None
            self.bm25_doc_ids = []

    def _save_bm25_to_disk(self) -> None:
        """Save BM25 retriever and id mapping to disk for persistent caching."""
        if self.bm25_retriever is None:
            return
        
        try:
            bm25_path = self.db_path / "bm25_index.pkl"
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25_retriever': self.bm25_retriever,
                    'bm25_doc_ids': self.bm25_doc_ids,
                }, f)
            logger.debug("BM25 index saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")

    def get_or_create_entities_collection(self) -> Any:
        """
        Get or create the entities collection.
        
        Returns:
            Chroma entities collection
        """
        self.entities_collection = self.client.get_or_create_collection(
            name=ENTITIES_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self.entities_collection

    def rebuild(self, chunks: Dict[str, Any], embeddings: Dict[str, List[float]]) -> None:
        """
        Rebuild knowledge base from content blocks and pre-calculated embeddings.
        
        This is a stateless operation that can be called independently.
        Also rebuilds BM25 index for full-text search.
        
        Args:
            chunks: Dict mapping chunk_id -> Chunk
            embeddings: Dict mapping chunk_id -> embedding vector
            collection_name: Name of collection to create/replace
        """
        try:
            # Delete existing collection
            self.client.delete_collection(COLLECTION_NAME)

            # Create new collection
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            embedding_list: List[List[float]] = []
            bm25_docs: List[str] = []
            
            for chunk_id, chunk in chunks.items():
                vector = embeddings.get(chunk_id)
                if vector is None:
                    raise RuntimeError(f"No embedding found for chunk {chunk_id}")

                ids.append(chunk_id)
                doc_text = chunk.text
                documents.append(doc_text)
                bm25_docs.append(doc_text)
                
                embedding_list.append(np.asarray(vector, dtype=float).tolist())
                metadatas.append({
                    "content_type": chunk.type.value if hasattr(chunk, 'type') else chunk.chunk_type,
                    "page_num": int(getattr(chunk, 'page_num', None) or 0),
                    "language": getattr(chunk, 'language', 'en'),
                    "source_file": str(getattr(chunk, 'source_file', '') or ""),
                })

            if not ids:
                raise RuntimeError("No content blocks to index into Chroma")

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embedding_list,
                metadatas=metadatas,
            )
            
            # Build BM25 index
            self._rebuild_bm25_index(bm25_docs, ids)
            
            logger.info(f"Chroma collection rebuilt with {len(ids)} chunks and BM25 indexed")

        except Exception as e:
            logger.error(f"Failed to rebuild collection: {e}")
            raise

    def _rebuild_bm25_index(self, documents: List[str], doc_ids: List[str]) -> None:
        """
        Rebuild BM25 index.
        
        Args:
            documents: List of document texts
            doc_ids: Corresponding document IDs
        """
        try:
            from ..retrieval.bm25_retriever import BM25Retriever
            from ..config import BM25Config
            
            config = BM25Config()
            self.bm25_retriever = BM25Retriever(config, documents)
            
            # Store ID mapping for retrieval
            self.bm25_doc_ids = doc_ids
            
            # Save to disk
            self._save_bm25_to_disk()
            
            logger.info(f"BM25 index built for {len(documents)} documents")
        except ImportError as e:
            logger.warning(f"BM25 module not available: {e}")
            self.bm25_retriever = None
            self.bm25_doc_ids = []
            raise
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_retriever = None
            self.bm25_doc_ids = []
            raise

    def build_bm25_index_from_chroma(self, name: str = COLLECTION_NAME) -> None:
        """
        Build BM25 index from the existing Chroma collection.
        
        Args:
            name: Chroma collection name
        """
        data = self.get_all(name)
        if not data:
            logger.warning("No Chroma data available to build BM25 index")
            raise RuntimeError("No Chroma data available to build BM25 index")

        ids = data.get("ids", [])
        documents = data.get("documents", [])

        if not ids or not documents:
            logger.warning("Chroma collection is empty; cannot build BM25 index")
            raise RuntimeError("Chroma collection is empty; cannot build BM25 index")

        self._rebuild_bm25_index(documents, ids)

    def rebuild_entities(self, entities: Dict[str, Any], embeddings: Dict[str, List[float]]) -> None:
        """
        Rebuild entities collection from entities and pre-calculated embeddings.
        
        Args:
            entities: Dict mapping entity_id -> Entity
            embeddings: Dict mapping entity_id -> embedding vector
        """
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(ENTITIES_COLLECTION_NAME)
            except:
                pass  # Collection might not exist

            # Create new collection
            self.entities_collection = self.client.get_or_create_collection(
                name=ENTITIES_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            embedding_list: List[List[float]] = []

            for entity_id, entity in entities.items():
                vector = embeddings.get(entity_id)
                if vector is None:
                    logger.warning(f"No embedding found for entity {entity_id}, skipping")
                    continue

                ids.append(entity_id)
                # Use entity name and description as document
                entity_name = getattr(entity, 'name', entity_id)
                entity_description = getattr(entity, 'description', '')
                documents.append(f"{entity_name}: {entity_description}")
                embedding_list.append(np.asarray(vector, dtype=float).tolist())
                metadatas.append({
                    "entity_type": getattr(entity, 'entity_type', 'unknown'),
                    "source_file": getattr(entity, 'source_file', ''),
                })

            if not ids:
                raise RuntimeError("No entities to index into Chroma")

            self.entities_collection.add(
                ids=ids,
                documents=documents,
                embeddings=embedding_list,
                metadatas=metadatas,
            )

            logger.info(f"Entities collection rebuilt with {len(ids)} entities")

        except Exception as e:
            logger.error(f"Failed to rebuild entities collection: {e}")
            raise

    def search(self, query_vector: List[float], top_k: int = 6, 
               collection_name: str = COLLECTION_NAME) -> List[RetrievalResult]:
        """
        Search for similar content blocks using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            collection_name: Collection to search in
            
        Returns:
            List of retrieval results ranked by score
        """
        if self.collection is None or self.collection.name != collection_name:
            self.get_or_create_collection(collection_name)

        try:
            raw = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )

            results: List[RetrievalResult] = []
            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            for block_id, content, meta, distance in zip(ids, docs, metas, distances):
                score = 1.0 - float(distance)
                score = max(0.0, min(1.0, score))
                results.append(
                    RetrievalResult(
                        doc_id=block_id,
                        score=score,
                        content=content,
                        content_type=ContentType(meta.get("content_type", "text")),
                        metadata={
                            **meta,
                            "retrieval_channel": "vector/chroma",
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            return []

    def search_bm25(self, query: str, top_k: int = 6) -> List[RetrievalResult]:
        """
        Search for relevant content using BM25 full-text search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of retrieval results ranked by BM25 score
        """
        if self.bm25_retriever is None:
            logger.warning("BM25 index not initialized, cannot perform BM25 search")
            return []
        
        try:
            # Get BM25 search results
            ranked_results = self.bm25_retriever.search(query, top_k=top_k)
            
            if not ranked_results:
                return []
            
            results: List[RetrievalResult] = []
            
            # Get the original documents from Chroma
            if self.collection is None:
                self.get_or_create_collection()
            
            # Map result indices to document IDs and retrieve from Chroma
            for doc_idx, bm25_score in ranked_results:
                if doc_idx >= len(self.bm25_doc_ids):
                    continue
                
                doc_id = self.bm25_doc_ids[doc_idx]
                
                # Get document from Chroma
                try:
                    data = self.collection.get(ids=[doc_id])
                    if data and data.get("documents"):
                        doc_text = data["documents"][0]
                        metadata = data.get("metadatas", [{}])[0] if data.get("metadatas") else {}
                        
                        # Normalize BM25 score to [0, 1]
                        # BM25 scores are typically in range [0, inf), we normalize them
                        normalized_score = min(1.0, bm25_score / (bm25_score + 1.0))
                        
                        results.append(
                            RetrievalResult(
                                doc_id=doc_id,
                                score=normalized_score,
                                content=doc_text,
                                content_type=ContentType(metadata.get("content_type", "text")),
                                metadata={
                                    **metadata,
                                    "retrieval_channel": "bm25/chroma",
                                    "bm25_raw_score": float(bm25_score),
                                },
                            )
                        )
                except Exception as e:
                    logger.debug(f"Failed to retrieve document {doc_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def count(self) -> int:
        """Get number of vectors in current collection."""
        if self.collection is None or self.collection.name != COLLECTION_NAME:
            self.get_or_create_collection(COLLECTION_NAME)
        return self.collection.count()

    def get_all(self, name: str = COLLECTION_NAME) -> Dict[str, Any]:
        """
        Get all data from the given collection.
        
        Returns:
            Dict with 'ids', 'documents', 'metadatas' keys
        """
        try:
            if self.collection is None or getattr(self.collection, "name", None) != name:
                collection_names = [
                    getattr(collection, "name", None)
                    for collection in self.client.list_collections()
                ]
                if name in collection_names:
                    self.collection = self.client.get_collection(name=name)
                else:
                    logger.warning(
                        "Chroma collection '%s' not found. Creating a new empty collection.",
                        name,
                    )
                    self.collection = self.get_or_create_collection(name)
            data = self.collection.get()
            ids = data.get("ids", [])
            if ids and isinstance(ids[0], list):
                ids = ids[0]

            documents = data.get("documents", [])
            if documents and isinstance(documents[0], list):
                documents = documents[0]

            metadatas = data.get("metadatas", [])
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]

            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "embeddings": data.get("embeddings") or [],
                "uris": data.get("uris") or [],
                "data": data.get("data") or [],
            }
        except Exception as e:
            logger.error(f"Failed to retrieve all data from Chroma collection '{name}': {e}")
    def search_entities(self, query_vector: List[float], top_k: int = 6) -> List[RetrievalResult]:
        """
        Search for similar entities using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of retrieval results ranked by score
        """
        if not hasattr(self, 'entities_collection') or self.entities_collection is None:
            self.get_or_create_entities_collection()

        try:
            raw = self.entities_collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )

            results: List[RetrievalResult] = []
            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            for entity_id, content, meta, distance in zip(ids, docs, metas, distances):
                score = 1.0 - float(distance)
                score = max(0.0, min(1.0, score))
                results.append(
                    RetrievalResult(
                        doc_id=entity_id,
                        score=score,
                        content=content,
                        content_type=ContentType.TEXT,  # Entities are text
                        metadata={
                            **meta,
                            "retrieval_channel": "vector/entities",
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Entities search failed: {e}")
            return []

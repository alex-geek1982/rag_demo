"""
Chroma Vector Knowledge Base - Isolated vector database operations
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

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
    - Independent rebuild from content blocks
    
    This class is completely decoupled from RAGEngine and other components.
    """


    def __init__(self, db_path: Path):
        """
        Initialize Chroma knowledge base.
        
        Args:
            db_path: Path to persistent Chroma database
        """
        self.db_path = db_path
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.collection = None
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise

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

            for chunk_id, chunk in chunks.items():
                vector = embeddings.get(chunk_id)
                if vector is None:
                    raise RuntimeError(f"No embedding found for chunk {chunk_id}")

                ids.append(chunk_id)
                documents.append(chunk.text)
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

        except Exception as e:
            logger.error(f"Failed to rebuild collection: {e}")
            raise

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

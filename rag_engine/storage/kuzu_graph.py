"""
Kuzu Graph Database - Isolated graph database operations
"""
import logging
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Default entity types for extraction
DEFAULT_ENTITY_TYPES = [
    "Person", "Creature", "Organization", "Location", "Event",
    "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"
]

from ..types import RetrievalResult, ContentType, Entity, Relationship, ContentBlock, ModalityType
from ..config import LLMConfig
from ..core.knowledge_graph import ExtractedEntity, ExtractedRelationship, LightRAGKnowledgeGraphBuilder
from ..core import prompts
from ..core.llm_client import get_llm_client

try:
    import networkx as nx
except ImportError:
    nx = None

logger = logging.getLogger(__name__)


@dataclass
class GraphEntity:
    """Entity for graph storage"""
    id: str
    name: str
    entity_type: str
    description: str
    source_block: str
    language: str = "zh"


@dataclass
class GraphRelationship:
    """Relationship for graph storage"""
    rel_id: str
    rel_type: str
    source_entity: str
    target_entity: str
    strength: float


class KuzuGraphStore:
    """
    Kuzu-based graph database for knowledge graph operations.
    
    Responsibilities:
    - Persist and manage entity and relationship data
    - Provide graph-based search and traversal
    - Independent rebuild from entity/relationship data or Chroma chunks
    
    This class is completely decoupled from RAGEngine.
    """

    def __init__(self, db_path: Path):
        """
        Initialize Kuzu graph store.
        
        Args:
            db_path: Path to Kuzu database directory
        """
        self.db_path = db_path
        self.content_blocks: Dict[str, ContentBlock] = {}
        try:
            import kuzu
            # Ensure directory exists and is not a file
            if db_path.exists():
                if db_path.is_file():
                    logger.warning(f"Removing file at {db_path} to create database directory")
                    db_path.unlink()
                elif not db_path.is_dir():
                    raise ValueError(f"Path {db_path} exists but is not a directory")
            db_path.mkdir(parents=True, exist_ok=True)
            self.db = kuzu.Database(str(db_path / "kuzu.db"))
            self.conn = kuzu.Connection(self.db)
            self._ensure_schema()
        except RuntimeError as e:
            if "Could not set lock" in str(e):
                logger.error(f"Database is locked: {db_path}. Another process may be using it. Close other instances and try again.")
                raise RuntimeError(f"Kuzu database is locked: {db_path}. Ensure no other processes are accessing it.") from e
            else:
                raise
        except ImportError:
            logger.error("kuzu not installed. Install with: pip install kuzu")
            raise

    def close(self) -> None:
        """Close the database connection."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _ensure_schema(self) -> None:
        """Ensure required tables exist."""
        try:
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS entities("
                "id STRING PRIMARY KEY, "
                "name STRING, "
                "entity_type STRING, "
                "description STRING, "
                "source_block STRING, "
                "language STRING, "
                "pagerank DOUBLE DEFAULT 0.0"
                ")"
            )
            self.conn.execute(
                "CREATE REL TABLE IF NOT EXISTS related("
                "FROM entities TO entities, "
                "rel_id STRING, "
                "rel_type STRING, "
                "strength DOUBLE"
                ")"
            )
        except Exception as e:
            logger.debug(f"Schema creation: {e}")

    @staticmethod
    def _escape(value: Any) -> str:
        """Escape string for Kuzu query."""
        return str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")

    def rebuild_from_entities_and_relationships(self, 
                                              entities: Dict[str, Entity],
                                              relationships: List,
                                              content_blocks: Dict[str, Any]) -> None:
        """
        Rebuild graph from entity and relationship data.
        
        This is the primary rebuild method when building from engine data.
        Handles both List[ExtractedRelationship] and Dict[str, Relationship].
        
        Args:
            entities: Dict of Entity objects
            relationships: List or Dict of Relationship/ExtractedRelationship objects
            content_blocks: Dict of content blocks for source_block mapping
        """
        self.content_blocks = content_blocks
        self._clear_and_recreate()

        # Convert dict to list if needed
        rel_list = relationships.values() if isinstance(relationships, dict) else relationships

        try:
            # Add entities
            for entity in entities.values():
                entity_id = getattr(entity, 'id', None) or getattr(entity, 'entity_name', None)
                entity_name = getattr(entity, 'name', None) or getattr(entity, 'entity_name', None)
                source_block = None
                if getattr(entity, 'source_blocks', None):
                    source_block = entity.source_blocks[0]
                else:
                    source_block = getattr(entity, 'source_id', None) or entity_id

                entity_type = getattr(entity, 'entity_type', None)
                description = getattr(entity, 'description', '') or ''
                language = 'zh'
                if hasattr(entity, 'attributes') and isinstance(entity.attributes, dict):
                    language = entity.attributes.get('language', language) or language
                elif hasattr(entity, 'language'):
                    language = getattr(entity, 'language', language) or language

                query = (
                    "CREATE (:entities {"
                    f"id: '{self._escape(entity_id)}', "
                    f"name: '{self._escape(entity_name)}', "
                    f"entity_type: '{self._escape(entity_type)}', "
                    f"description: '{self._escape(description)}', "
                    f"source_block: '{self._escape(source_block)}', "
                    f"language: '{self._escape(language)}', "
                    "pagerank: 0.0"
                    "})"
                )
                self.conn.execute(query)

            # Add relationships
            for rel in rel_list:
                # Handle both ExtractedRelationship (src_id/tgt_id) and Relationship (source_entity/target_entity)
                src_id = getattr(rel, 'src_id', None) or getattr(rel, 'source_entity', None)
                tgt_id = getattr(rel, 'tgt_id', None) or getattr(rel, 'target_entity', None)
                rel_type = getattr(rel, 'keywords', None) or getattr(rel, 'relationship_type', 'related')
                strength = getattr(rel, 'weight', None) or getattr(rel, 'strength', 1.0)
                rel_id = getattr(rel, 'id', f"rel_{src_id}_to_{tgt_id}")

                if src_id not in entities or tgt_id not in entities:
                    continue

                query = (
                    f"MATCH (a:entities {{id:'{self._escape(src_id)}'}}), "
                    f"(b:entities {{id:'{self._escape(tgt_id)}'}}) "
                    "CREATE (a)-[:related {"
                    f"rel_id:'{self._escape(rel_id)}', "
                    f"rel_type:'{self._escape(rel_type)}', "
                    f"strength:{float(strength)}"
                    "}]->(b)"
                )
                self.conn.execute(query)

            # Compute PageRank
            self._compute_pagerank()

            logger.info(f"Graph rebuilt with {len(entities)} entities and {len(rel_list)} relationships")

        except Exception as e:
            logger.error(f"Failed to rebuild graph from entities: {e}")
            raise

    def rebuild_from_chroma_chunks(self, chunk_ids: List[str], documents: List[str], 
                                   metadatas: List[Dict[str, Any]],
                                   llm_config: Optional[LLMConfig] = None, global_config: Dict[str, Any] = None) -> None:
        """
        Rebuild graph from Chroma knowledge base chunks using LightRAG pipeline.
        
        This is a stateless operation that leverages the full gleaning + merging pipeline
        to build high-quality knowledge graphs from existing KB chunks.
        
        Args:
            chunk_ids: List of chunk IDs from Chroma
            documents: List of chunk content texts
            metadatas: List of chunk metadata dicts
            llm_config: LLMConfig for model selection (uses default if not provided)
            global_config: Global configuration for KG building
        """
        # Use LightRAG pipeline: convert chunks → ContentBlocks → build_from_content_blocks
        logger.info(f"Rebuilding graph from {len(chunk_ids)} Chroma chunks using LightRAG pipeline")
        
        # Convert Chroma chunks to ContentBlocks
        content_blocks = self._convert_chroma_chunks_to_content_blocks(
            chunk_ids, documents, metadatas
        )
        
        # Use full LightRAG pipeline
        try:
            # Initialize KG builder
            kg_builder = LightRAGKnowledgeGraphBuilder(
                language="en",
                llm_config=llm_config,
                global_config=global_config or {}
            )
            
            # Run async pipeline
            asyncio.run(
                kg_builder.build_from_content_blocks(content_blocks)
            )
            
            # Save to Kuzu database
            self.rebuild_from_extracted_data(kg_builder.entities, kg_builder.relationships)
            
            logger.info(
                f"Graph rebuilt from {len(chunk_ids)} Chroma chunks: "
                f"{len(kg_builder.entities)} entities, {len(kg_builder.relationships)} relationships"
            )
            
        except Exception as e:
            logger.error(f"Failed to rebuild graph from Chroma chunks with LightRAG: {e}")
            # Fallback to simple approach
            logger.info("Falling back to simple chunk-entity mapping...")
            self._rebuild_from_chroma_chunks_simple(chunk_ids, documents, metadatas)
    
    def _rebuild_from_chroma_chunks_simple(self, chunk_ids: List[str], documents: List[str],
                                          metadatas: List[Dict[str, Any]]) -> None:
        """
        Simple fallback: Create entities for each chunk with sequential relationships.
        Used when LLM function is not available.
        """
        self._clear_and_recreate()
        
        try:
            # Create entities for each chunk
            for chunk_id, document, metadata in zip(chunk_ids, documents, metadatas):
                content_type = metadata.get("content_type", "text")
                language = metadata.get("language", "zh")

                query = (
                    "CREATE (:entities {"
                    f"id: '{self._escape(chunk_id)}', "
                    f"name: '{self._escape(chunk_id)}', "
                    f"entity_type: '{self._escape(content_type)}', "
                    f"description: '{self._escape(document[:200])}', "
                    f"source_block: '{self._escape(chunk_id)}', "
                    f"language: '{self._escape(language)}'"
                    "})"
                )
                self.conn.execute(query)

            # Create sequential relationships between chunks
            for i in range(len(chunk_ids) - 1):
                current_id = chunk_ids[i]
                next_id = chunk_ids[i + 1]

                query = (
                    f"MATCH (a:entities {{id:'{self._escape(current_id)}'}}), "
                    f"(b:entities {{id:'{self._escape(next_id)}'}}) "
                    "CREATE (a)-[:related {"
                    f"rel_id: 'rel_{i}_to_{i + 1}', "
                    f"rel_type: 'follows', "
                    f"strength: 0.7"
                    "}]->(b)"
                )
                self.conn.execute(query)

            logger.info(f"Graph rebuilt from {len(chunk_ids)} Chroma chunks (simple mode)")

        except Exception as e:
            logger.error(f"Failed to rebuild graph from Chroma chunks (simple mode): {e}")
            raise
    
    def _convert_chroma_chunks_to_content_blocks(self, chunk_ids: List[str], 
                                                documents: List[str],
                                                metadatas: List[Dict[str, Any]]) -> List[ContentBlock]:
        """
        Convert Chroma chunks to ContentBlock format for LightRAG pipeline.
        
        Args:
            chunk_ids: List of chunk IDs
            documents: List of chunk content texts
            metadatas: List of chunk metadata
            
        Returns:
            List of ContentBlock objects
        """
        content_blocks = []
        
        for chunk_id, document, metadata in zip(chunk_ids, documents, metadatas):
            # Determine content type
            content_type_str = metadata.get("content_type", "text")
            try:
                content_type = ContentType(content_type_str)
            except (ValueError, KeyError):
                content_type = ContentType.TEXT
            
            # Create ContentBlock
            block = ContentBlock(
                id=chunk_id,
                type=content_type,
                modality=ModalityType.TEXT,  # Chunks are typically text
                content=document,
                metadata=metadata,
                source_file=metadata.get("source_file", "chroma_kb"),
                page_num=metadata.get("page_num"),
                language=metadata.get("language", "en")
            )
            content_blocks.append(block)
        
        logger.debug(f"Converted {len(content_blocks)} Chroma chunks to ContentBlocks")
        return content_blocks

    def _clear_and_recreate(self) -> None:
        """Delete and recreate graph tables."""
        try:
            self.conn.execute("DROP TABLE IF EXISTS related")
            self.conn.execute("DROP TABLE IF EXISTS entities")
            self._ensure_schema()
        except Exception as e:
            logger.debug(f"Clear and recreate: {e}")

    def _compute_pagerank(self) -> None:
        """Compute PageRank for all entities and update the database."""
        if nx is None:
            logger.warning("networkx not installed, skipping PageRank calculation")
            return

        try:
            # Get all entities
            query = "MATCH (e:entities) RETURN e.id"
            df = self.conn.execute(query).get_as_df()
            nodes = df['e.id'].tolist()

            if not nodes:
                return

            # Get all relationships
            query = "MATCH (a:entities)-[r:related]->(b:entities) RETURN a.id, b.id, r.strength"
            df = self.conn.execute(query).get_as_df()
            edges = [(row['a.id'], row['b.id'], float(row['r.strength'])) for _, row in df.iterrows()]

            # Build graph
            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            for src, tgt, weight in edges:
                G.add_edge(src, tgt, weight=weight)

            # Compute PageRank
            pagerank = nx.pagerank(G, weight='weight')

            # Update entities
            for node, pr in pagerank.items():
                update_query = f"MATCH (e:entities) WHERE e.id = '{self._escape(node)}' SET e.pagerank = {pr}"
                self.conn.execute(update_query)

            logger.info(f"Computed PageRank for {len(nodes)} entities")

        except Exception as e:
            logger.error(f"Failed to compute PageRank: {e}")

    async def search(self, query_entities: List[RetrievalResult], top_k: int = 6, n_hop: int = 2) -> Dict[str, Any]:
        """
        Search graph directly from Kuzu without external content_blocks dependency.
        
        Performs structured retrieval of entities and relationships from the knowledge graph,
        combining keyword matching, PageRank scoring, and n-hop traversal.
        
        Args:
            query_entities: 
            top_k: Maximum number of entities to return
            n_hop: Number of hops for relationship traversal
            
        Returns:
            Dict containing structured results:
            {
                "entities": [
                    {
                        "name": str,
                        "entity_type": str,
                        "description": str,
                        "score": float,
                        "pagerank": float
                    },
                    ...
                ],
                "relationships": [
                    {
                        "from_entity": str,
                        "to_entity": str,
                        "rel_type": str,
                        "strength": float,
                        "score": float
                    },
                    ...
                ]
            }
        """
        entity_scores: Dict[str, Dict[str, Any]] = {}
        relationship_scores: Dict[tuple, Dict[str, Any]] = {}

        # ===== Step 1: Get all entities and match with keywords =====
        # Convert List[RetrievalResult] to dict format
        for result in query_entities:
            entity_scores[result.doc_id] = {
                "score": result.score,
                "pagerank": result.metadata.get("pagerank", 0.1),
                "entity_type": result.metadata.get("entity_type", "Unknown"),
                "description": result.content,
            }

        logger.info(f"Found {len(entity_scores)} entities matching query")

        # ===== Step 2: Get relationships for top entities =====
        top_entity_ids = [e_id for e_id in sorted(
            entity_scores.keys(),
            key=lambda x: entity_scores[x]["score"] * entity_scores[x]["pagerank"],
            reverse=True
        )[:max(top_k * 2, 4)]]

        query_relationships = self._get_relationships_for_entities(top_entity_ids)
        relationship_scores.update(query_relationships)

        logger.info(f"Found {len(relationship_scores)} relationships")

        # ===== Step 3: Get n-hop neighbors and their relationships =====
        for entity_id in top_entity_ids:
            nhop_ents = self._get_nhop_neighbors(entity_id, n_hop)
            entity_scores.update(nhop_ents)

            # Also get relationships involving n-hop neighbors
            nhop_rels = self._get_relationships_for_entities([e_id for e_id in nhop_ents.keys()])
            relationship_scores.update(nhop_rels)

        logger.info(f"Expanded to {len(entity_scores)} entities via n-hop traversal")

        # ===== Step 4: Sort and format results =====
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1]["score"] * x[1]["pagerank"],
            reverse=True
        )[:top_k]

        sorted_relationships = sorted(
            relationship_scores.items(),
            key=lambda x: x[1]["strength"] * x[1]["score"],
            reverse=True
        )[:top_k]

        # Format output
        entities_output = [
            {
                "Entity": entity_id,
                "Type": entity_data["entity_type"],
                "Description": entity_data["description"],
                "Score": round(entity_data["score"] * entity_data["pagerank"], 4),
                "PageRank": round(entity_data["pagerank"], 4),
            }
            for entity_id, entity_data in sorted_entities
        ]

        relationships_output = [
            {
                "From Entity": rel_data["from_entity"],
                "To Entity": rel_data["to_entity"],
                "Type": rel_data["rel_type"],
                "Strength": round(rel_data["strength"], 4),
                "Score": round(rel_data["score"], 4),
            }
            for (from_ent, to_ent), rel_data in sorted_relationships
        ]

        return {
            "entities": entities_output,
            "relationships": relationships_output,
            "total_entities": len(entity_scores),
            "total_relationships": len(relationship_scores),
        }

    async def _get_relevant_entities(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Get entities matching query keywords from graph database.

        Uses LLM-based entity extraction via ENTITY_EXTRACTION_USER_PROMPT instead of hard-coded rules.

        Returns:
            Dict mapping entity_id to entity info with score
        """
        entities = {}

        if not query:
            return entities

        # Format system and user prompts correctly
        system_prompt = prompts.get_system_prompt(entity_types=DEFAULT_ENTITY_TYPES)
        user_prompt = prompts.ENTITY_EXTRACTION_USER_PROMPT.format(
            input_text=query,
            tuple_delimiter=prompts.TUPLE_DELIMITER,
            completion_delimiter=prompts.COMPLETION_DELIMITER
        )

        llm_client = get_llm_client(LLMConfig())
        response = await llm_client.generate_with_system(system_prompt, user_prompt)

        extracted_entities = self._parse_llm_query_entities(response)
        if not extracted_entities:
            logger.debug("No entities extracted from query via LLM prompt")
            return entities

        for entity_name, entity_type, description in extracted_entities:
            query = (
                "MATCH (e:entities) "
                f"WHERE e.name = '{self._escape(entity_name)}' OR e.id = '{self._escape(entity_name)}' "
                "RETURN e.id, e.name, e.entity_type, e.description, e.pagerank "
                "LIMIT 1"
            )
            df = self.conn.execute(query).get_as_df()
            if df.empty:
                logger.debug(f"LLM-extracted entity '{entity_name}' not found in Kuzu graph")
                continue

            row = df.iloc[0]
            entity_id = row['e.id']
            entities[entity_id] = {
                "name": row['e.name'],
                "entity_type": row['e.entity_type'],
                "description": row['e.description'] or description,
                "score": 1.0,
                "pagerank": float(row.get('e.pagerank', 1.0))
            }

        return entities

    def _parse_llm_query_entities(self, response: str) -> List[tuple]:
        """
        Parse entity lines returned from the LLM query extraction prompt.

        Returns tuples of (entity_name, entity_type, description).
        """
        extracted = []
        lines = response.strip().splitlines()

        for line in lines:
            if prompts.COMPLETION_DELIMITER in line:
                break

            line = line.strip()
            if not line or not line.startswith('entity'):
                continue

            try:
                if prompts.TUPLE_DELIMITER in line:
                    parts = line.split(prompts.TUPLE_DELIMITER)
                    if len(parts) >= 4 and parts[0].strip() == 'entity':
                        entity_name = parts[1].strip()
                        entity_type = parts[2].strip()
                        entity_description = parts[3].strip()
                        extracted.append((entity_name, entity_type, entity_description))
                        continue

                # Fallback for legacy delimiter format
                match = re.match(r"entity<\|#\|>([^<]+)<\|#\|>([^<]+)<\|#\|>(.+)", line)
                if match:
                    entity_name = match.group(1).strip()
                    entity_type = match.group(2).strip()
                    entity_description = match.group(3).strip()
                    extracted.append((entity_name, entity_type, entity_description))
            except Exception as e:
                logger.warning(f"Failed to parse LLM entity line: {line}, error: {e}")
                continue

        return extracted

    def _get_nhop_neighbors(self, entity_id: str, n_hop: int) -> Dict[str, Dict[str, Any]]:
        """
        Get n-hop neighbors of an entity from graph.
        
        Args:
            entity_id: Starting entity ID
            n_hop: Number of hops
            
        Returns:
            Dict mapping neighbor entity_id to entity info
        """
        neighbors = {}

        try:
            # n-hop query using breadth-first traversal
            neighbor_query = (
                f"MATCH (e:entities {{id:'{self._escape(entity_id)}'}})-[*1..{n_hop}]-(n:entities) "
                "WHERE e <> n "
                "RETURN DISTINCT n.id, n.name, n.entity_type, n.description, n.pagerank"
            )
            df = self.conn.execute(neighbor_query).get_as_df()

            for _, row in df.iterrows():
                neighbor_id = row["n.id"]
                pagerank = float(row.get("n.pagerank", 0.0))
                
                # n-hop neighbors get lower base score, but retain their PageRank
                distance_factor = 0.6 / (n_hop + 1)
                sim_score = min(0.95, distance_factor)

                neighbors[neighbor_id] = {
                    "name": row["n.name"],
                    "entity_type": row["n.entity_type"],
                    "description": row["n.description"] or "",
                    "score": sim_score,
                    "pagerank": pagerank,
                }

            logger.debug(f"Retrieved {len(neighbors)} n-hop neighbors for {entity_id} (hops={n_hop})")
            return neighbors

        except Exception as e:
            logger.error(f"Error retrieving n-hop neighbors for {entity_id}: {e}")
            return neighbors

    def _get_relationships_for_entities(self, entity_ids: List[str]) -> Dict[tuple, Dict[str, Any]]:
        """
        Get relationships between and involving given entities.
        
        Args:
            entity_ids: List of entity IDs
            
        Returns:
            Dict mapping (from_entity, to_entity) to relationship info
        """
        relationships = {}

        if not entity_ids:
            return relationships

        try:
            # Get all relationships involving these entities
            for entity_id in entity_ids:
                # Outgoing relationships
                out_query = (
                    f"MATCH (a:entities {{id:'{self._escape(entity_id)}'}})-[r:related]->(b:entities) "
                    "RETURN a.id, b.id, b.name, r.rel_type, r.strength"
                )
                out_df = self.conn.execute(out_query).get_as_df()

                for _, row in out_df.iterrows():
                    from_ent = row["a.id"]
                    to_ent = row["b.id"]
                    rel_type = row.get("r.rel_type", "related")
                    strength = float(row.get("r.strength", 1.0))

                    key = (from_ent, to_ent)
                    relationships[key] = {
                        "from_entity": from_ent,
                        "to_entity": to_ent,
                        "rel_type": rel_type,
                        "strength": strength,
                        "score": strength,  # Base score is the relationship strength
                    }

                # Incoming relationships
                in_query = (
                    f"MATCH (a:entities)-[r:related]->(b:entities {{id:'{self._escape(entity_id)}'}}) "
                    "RETURN a.id, a.name, b.id, r.rel_type, r.strength"
                )
                in_df = self.conn.execute(in_query).get_as_df()

                for _, row in in_df.iterrows():
                    from_ent = row["a.id"]
                    to_ent = row["b.id"]
                    rel_type = row.get("r.rel_type", "related")
                    strength = float(row.get("r.strength", 1.0))

                    key = (from_ent, to_ent)
                    if key not in relationships:
                        relationships[key] = {
                            "from_entity": from_ent,
                            "to_entity": to_ent,
                            "rel_type": rel_type,
                            "strength": strength,
                            "score": strength,
                        }

            logger.debug(f"Retrieved {len(relationships)} relationships for {len(entity_ids)} entities")
            return relationships

        except Exception as e:
            logger.error(f"Error retrieving relationships: {e}")
            return relationships

    def get_all_entities(self) -> Dict[str, Entity]:
        """
        Get all entities from the graph database.
        
        Returns:
            Dict of entity_id to Entity objects
        """
        try:
            query = "MATCH (e:entities) RETURN e.id, e.name, e.entity_type, e.description, e.source_block, e.language"
            df = self.conn.execute(query).get_as_df()
            
            entities = {}
            for _, row in df.iterrows():
                entity = Entity(
                    id=row['e.id'],
                    name=row['e.name'],
                    entity_type=row['e.entity_type'],
                    description=row['e.description'],
                    source_blocks=[row['e.source_block']],
                    attributes={'language': row['e.language']}
                )
                entities[row['e.id']] = entity
            
            return entities
        except Exception as e:
            logger.error(f"Failed to get all entities: {e}")
            return {}

    @staticmethod
    def _extract_query_terms(query: str) -> List[str]:
        """Extract searchable terms from query."""
        terms = set()

        # English tokens
        for token in re.findall(r"[A-Za-z0-9_]{2,}", query.lower()):
            terms.add(token)

        # Chinese bi-grams and n-grams
        for zh_segment in re.findall(r"[\u4e00-\u9fff]{2,}", query):
            terms.add(zh_segment)
            for n in (2, 3, 4):
                for i in range(0, max(0, len(zh_segment) - n + 1)):
                    terms.add(zh_segment[i : i + n])

        return sorted(t for t in terms if len(t.strip()) >= 2)

    def rebuild_from_extracted_data(self, entities: Dict[str, ExtractedEntity], 
                                   relationships: List[ExtractedRelationship]) -> None:
        """
        Rebuild graph from LightRAG-style extracted entity and relationship data.
        
        Args:
            entities: Dict of ExtractedEntity objects
            relationships: List of ExtractedRelationship objects
        """
        self._clear_and_recreate()

        try:
            # Build a mapping of entity names for relationship resolution
            entity_name_to_id = {}
            
            # Add entities
            for entity_key, entity in entities.items():
                entity_id = entity.entity_name  # Use entity_name as the actual ID
                entity_name_to_id[entity_key] = entity_id  # Map from dict key to entity_name
                
                query = (
                    "CREATE (:entities {"
                    f"id: '{self._escape(entity_id)}', "
                    f"name: '{self._escape(entity.entity_name)}', "
                    f"entity_type: '{self._escape(entity.entity_type)}', "
                    f"description: '{self._escape(entity.description)}', "
                    f"source_block: '{self._escape(entity.source_id)}', "
                    f"language: 'zh'"
                    "})"
                )
                self.conn.execute(query)

            # Add relationships using entity names as IDs
            for rel in relationships:
                # Map src_id and tgt_id using the entity name mapping
                src_entity_id = entity_name_to_id.get(rel.src_id, rel.src_id)
                tgt_entity_id = entity_name_to_id.get(rel.tgt_id, rel.tgt_id)
                
                # Verify both entities exist
                if src_entity_id not in [entity_name_to_id.get(k, k) for k in entities.keys()]:
                    logger.debug(f"Source entity {src_entity_id} not found, skipping relationship")
                    continue
                if tgt_entity_id not in [entity_name_to_id.get(k, k) for k in entities.keys()]:
                    logger.debug(f"Target entity {tgt_entity_id} not found, skipping relationship")
                    continue

                query = (
                    f"MATCH (a:entities {{id:'{self._escape(src_entity_id)}'}}), "
                    f"(b:entities {{id:'{self._escape(tgt_entity_id)}'}}) "
                    "CREATE (a)-[:related {"
                    f"rel_id:'rel_{self._escape(src_entity_id)}_to_{self._escape(tgt_entity_id)}', "
                    f"rel_type:'{self._escape(rel.keywords or 'related')}', "
                    f"strength:{float(rel.weight)}"
                    "}]->(b)"
                )
                self.conn.execute(query)

            # Compute PageRank for the newly built graph
            self._compute_pagerank()

            logger.info(f"Graph rebuilt with {len(entities)} entities and {len(relationships)} relationships (LightRAG style)")

        except Exception as e:
            logger.error(f"Failed to rebuild graph from extracted data: {e}")
            raise

"""
Knowledge Graph Builder - Build and manage knowledge graph
"""
import logging
import asyncio
from typing import Dict, List, Optional

from ..config import RAGEngineConfig, LLMConfig
from ..core.knowledge_graph import (
    KnowledgeGraph,
    EntityExtractor,
    RelationshipBuilder,
    LightRAGKnowledgeGraphBuilder,
)
from ..types import Chunk, Document, ContentBlock, Entity, Relationship
from ..storage import KuzuGraphStore
from ..retrieval import OpenAIEmbedding

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Responsibility: Build and manage the knowledge graph.
    
    This class handles:
    - Entity extraction from documents
    - Relationship building between entities
    - Knowledge graph persistence
    - Independent rebuild from documents, blocks, or Chroma data
    
    It is independent of document processing or query execution.
    """

    def __init__(self, config: RAGEngineConfig):
        """
        Initialize knowledge graph builder.
        
        Args:
            config: RAG engine configuration
        """
        self.config = config
        self.kg = None
        self.entity_extractor = None
        self.relationship_builder = None
        self.entity_embedding_provider = None
        self.entity_embeddings: Dict[str, List[float]] = {}

    def _init_kg(self, llm_config: Optional[LLMConfig] = None, global_config=None) -> None:
        """Initialize knowledge graph (lazy)."""
        if self.kg is not None:
            return

        self.kg = KnowledgeGraph()
        # Use provided config or create default from engine config
        if llm_config is None:
            llm_config = LLMConfig(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url
            )
        
        self.entity_extractor = EntityExtractor(
            self.config.language.default_language,
            llm_config=llm_config,
            global_config=global_config
        )
        self.relationship_builder = RelationshipBuilder(
            self.config.language.default_language,
            llm_config=llm_config,
            global_config=global_config
        )

    def _init_entity_embeddings(self) -> None:
        """Initialize entity embedding provider (lazy)."""
        if self.entity_embedding_provider is not None:
            return

        if not self.config.embedding.api_key:
            raise ValueError("OpenAI API key not configured for embeddings")

        self.entity_embedding_provider = OpenAIEmbedding(
            api_key=self.config.embedding.api_key,
            model=self.config.embedding.model,
            base_url=self.config.embedding.base_url,
            use_azure=self.config.embedding.use_azure,
            azure_endpoint=self.config.embedding.azure_endpoint,
            azure_api_version=self.config.embedding.azure_api_version,
            azure_deployment=self.config.embedding.azure_deployment,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count approximate tokens in text.
        
        Uses a simple approximation: 1 token ≈ 4 characters
        This is a rough estimate. For precise token counting, use tiktoken or the LLM provider's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        if not text:
            return 0
        # Simple approximation: ~4 characters per token (OpenAI default)
        return max(1, len(text) // 4)

    # ========== 步骤 1: 通过LLM提取实体和关系 ==========
    
    async def extract_entities_and_relationships_async(self, blocks: List[ContentBlock], llm_config: Optional[LLMConfig] = None, global_config=None) -> tuple:
        """
        步骤 1: 通过LLM提取实体和实体关系（无存储操作）
        
        Optimized version: Extracts entities and relationships in a single LLM call
        instead of two separate calls, reducing API costs and improving performance.
        
        This step is completely decoupled from storage. It only:
        - Extracts entities using LLM
        - Builds relationships using LLM (in the same request)
        - Stores extracted data in memory (self.kg)
        
        Args:
            blocks: List of content blocks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
            
        Returns:
            Tuple of (entities list, relationships list)
        """
        logger.info(f"[Step 1] 通过单次LLM请求提取实体和关系，从 {len(blocks)} 个块")

        try:
            self._init_kg(llm_config=llm_config, global_config=global_config)

            # Extract both entities and relationships in a single LLM call
            logger.info("  提取实体和关系...")
            entities, relationships = await self.entity_extractor.extract_entities_and_relationships_in_one_call(blocks)
            logger.info(f"  ✓ 已提取 {len(entities)} 个实体和 {len(relationships)} 个关系")

            # Add entities to in-memory graph
            for entity in entities:
                self.kg.add_entity(entity)

            # Add relationships to in-memory graph
            for rel in relationships:
                self.kg.add_relationship(rel)

            return entities, relationships

        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {e}")
            raise

    def extract_entities_and_relationships(self, blocks: List[ContentBlock], llm_config: Optional[LLMConfig] = None, global_config=None) -> tuple:
        """
        步骤 1 (同步包装): 通过LLM提取实体和关系
        
        Args:
            blocks: List of content blocks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
            
        Returns:
            Tuple of (entities list, relationships list)
        """
        return asyncio.run(self.extract_entities_and_relationships_async(blocks, llm_config, global_config))

    # ========== 步骤 2: 实体向量化并插入Chroma ==========
    
    def merge_chunks_by_token_size(self, chunks: List, max_tokens: int = 4096) -> List:
        """
        Merge chunks based on token count.
        
        Rules:
        1. Do NOT split chunks > max_tokens (keep as-is)
        2. Do NOT force merging to reach max_tokens (if current + next >= max_tokens, don't merge)
        
        Args:
            chunks: List of Chunk objects to merge
            max_tokens: Maximum tokens per merged chunk (default 4096)
            
        Returns:
            List of merged Chunk objects
        """
        if not chunks:
            return chunks
        
        merged = []
        
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk.text)
            
            # Rule 1: If chunk > max_tokens, don't split - add as standalone
            if chunk_tokens > max_tokens:
                merged.append(chunk)
                continue
            
            # Try to merge with previous chunk
            if merged:
                last_chunk = merged[-1]
                last_tokens = self._count_tokens(last_chunk.text)
                total_tokens = last_tokens + chunk_tokens
                
                # Rule 2: Only merge if total tokens < max_tokens
                if total_tokens < max_tokens:
                    # Merge by concatenating text
                    merged[-1].text += '\n' + chunk.text
                    # Update metadata with combined info
                    merged[-1].metadata['merged'] = True
                    merged[-1].metadata['merged_source_blocks'] = (
                        merged[-1].metadata.get('merged_source_blocks', merged[-1].source_block_ids) +
                        chunk.source_block_ids
                    )
                    continue
            
            # Add as new chunk
            merged.append(chunk)
        
        logger.debug(f"Merged {len(chunks)} chunks into {len(merged)} chunks (max {max_tokens} tokens)")
        return merged
    
    def embed_entities_and_store_to_chroma(self, entities: List[Entity], chroma_kb) -> None:
        """
        步骤 2: 实体向量化并插入Chroma
        
        This step:
        - Generates embeddings for extracted entities
        - Stores entity vectors in Chroma
        
        Args:
            entities: List of extracted entities
            chroma_kb: ChromaKnowledgeBase instance for storage
        """
        logger.info(f"[Step 2] 实体向量化并存储到Chroma")

        try:
            # Generate embeddings for entities
            logger.info(f"  为 {len(entities)} 个实体生成向量...")
            self._generate_entity_embeddings(entities)
            logger.info(f"  ✓ 已生成 {len(self.entity_embeddings)} 个实体向量")

            # Store entities and embeddings to Chroma
            logger.info("  将实体向量存储到Chroma...")
            if self.kg is not None:
                chroma_kb.rebuild_entities(self.kg.entities, self.entity_embeddings)
            logger.info("  ✓ 实体已存储到Chroma")

        except Exception as e:
            logger.error(f"Failed to embed and store entities to Chroma: {e}")
            raise

    # ========== 步骤 3: 实体和关系插入图数据库 ==========
    
    def store_entities_and_relationships_to_kuzu(self, kuzu_store: KuzuGraphStore, content_blocks: Dict[str, ContentBlock] = None) -> None:
        """
        步骤 3: 实体和关系插入Kuzu图数据库
        
        This step:
        - Stores entities in Kuzu
        - Stores relationships in Kuzu
        
        Args:
            kuzu_store: KuzuGraphStore instance for storage
            content_blocks: Optional dict mapping block IDs to ContentBlock objects (for source tracking)
        """
        logger.info(f"[Step 3] 实体和关系存储到Kuzu图数据库")

        try:
            if self.kg is None:
                raise RuntimeError("No knowledge graph available. Run step 1 first.")

            if content_blocks is None:
                content_blocks = {}

            # Store entities and relationships to Kuzu
            logger.info(f"  存储 {len(self.kg.entities)} 个实体和 {len(self.kg.relationships)} 个关系到Kuzu...")
            kuzu_store.rebuild_from_entities_and_relationships(
                self.kg.entities, 
                self.kg.relationships, 
                content_blocks
            )
            logger.info("  ✓ 实体和关系已存储到Kuzu")

        except Exception as e:
            logger.error(f"Failed to store entities and relationships to Kuzu: {e}")
            raise

    # ========== 向后兼容的包装方法 ==========

    def build_from_document(self, document: Document, llm_config: Optional[LLMConfig] = None, global_config=None) -> None:
        """
        Build knowledge graph from a single document.
        
        This method:
        1. Converts chunks to content blocks
        2. Extracts entities and relationships
        3. Stores them in knowledge graph
        
        Args:
            document: Document with chunks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
        """
        logger.info(f"Building KG from document: {document.title}")
        from ..types import chunks_to_content_blocks
        
        # Convert chunks to content blocks for KG building
        content_blocks = chunks_to_content_blocks(
            document.chunks, 
            source_file=document.source_path,
            language=document.language
        )
        
        asyncio.run(self.build_from_blocks_async(content_blocks, llm_config, global_config))

    async def build_from_blocks_async(self, blocks: List[ContentBlock], llm_config: Optional[LLMConfig] = None, global_config=None) -> None:
        """
        Build knowledge graph from content blocks (legacy wrapper, async).
        
        This method combines all three steps for backward compatibility.
        
        Args:
            blocks: List of content blocks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
        """
        logger.info(f"Building KG from {len(blocks)} blocks (legacy mode)")

        try:
            # Step 1: Extract entities and relationships
            entities, relationships = await self.extract_entities_and_relationships_async(
                blocks, llm_config, global_config
            )
            
            # Step 2: Generate entity embeddings (for legacy compatibility)
            logger.info("Generating embeddings for entities...")
            self._generate_entity_embeddings(entities)

        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            raise

    def _generate_entity_embeddings(self, entities: List[Entity]) -> None:
        """Generate embeddings for entities."""
        if not entities:
            return

        try:
            self._init_entity_embeddings()

            # Prepare texts for embedding
            entity_texts = []
            entity_ids = []
            for entity in entities:
                # Use entity_name.lower() as the key to match how entities are stored in KnowledgeGraph
                entity_name = getattr(entity, 'entity_name', getattr(entity, 'name', ''))
                if not entity_name:
                    logger.warning(f"Entity has no name attribute, skipping: {entity}")
                    continue
                entity_id = entity_name.lower()
                entity_description = getattr(entity, 'description', '')
                text = f"{entity_name}: {entity_description}"
                entity_texts.append(text)
                entity_ids.append(entity_id)

            # Generate embeddings in batches
            logger.info(f"Generating embeddings for {len(entity_texts)} entities...")
            embedding_vectors = self.entity_embedding_provider.embed_text_sync(entity_texts)

            # Map embeddings to entity IDs
            for entity_id, vector in zip(entity_ids, embedding_vectors):
                self.entity_embeddings[entity_id] = vector.tolist()

            logger.info(f"Generated {len(self.entity_embeddings)} entity embeddings")

        except Exception as e:
            logger.error(f"Failed to generate entity embeddings: {e}")
            raise

    def build_from_blocks(self, blocks: List[ContentBlock], llm_config: Optional[LLMConfig] = None, global_config=None) -> None:
        """
        Build knowledge graph from content blocks (synchronous wrapper).
        
        Args:
            blocks: List of content blocks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
        """
        asyncio.run(self.build_from_blocks_async(blocks, llm_config, global_config))

    def rebuild_kuzu(self, kuzu_store: KuzuGraphStore) -> None:
        """
        Rebuild Kuzu graph database with current KG data.
        
        Args:
            kuzu_store: KuzuGraphStore instance
        """
        if self.kg is None:
            raise RuntimeError("No knowledge graph available. Call build_from_blocks first.")

        logger.info("Rebuilding Kuzu graph database...")
        # Note: For rebuild_kuzu, we need content_blocks to map source_blocks
        # This is passed separately in the example
        kuzu_store.rebuild_from_entities_and_relationships(
            self.kg.entities, self.kg.relationships, {}
        )

    def rebuild_kuzu_from_chroma_chunks(
        self,
        kuzu_store: KuzuGraphStore,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        """
        Rebuild Kuzu graph database from Chroma knowledge base chunks.
        
        This is a stateless operation that doesn't require document processing.
        Useful for independently rebuilding knowledge graph from existing KB.
        
        Args:
            kuzu_store: KuzuGraphStore instance
            chunk_ids: List of chunk IDs from Chroma
            documents: List of chunk documents from Chroma
            metadatas: List of chunk metadatas from Chroma
        """
        logger.info(f"Rebuilding Kuzu graph from {len(chunk_ids)} Chroma chunks...")
        kuzu_store.rebuild_from_chroma_chunks(chunk_ids, documents, metadatas)

    def get_entities(self) -> Dict[str, Entity]:
        """Get all entities from knowledge graph."""
        if self.kg is None:
            return {}
        return self.kg.entities.copy()

    def get_relationships(self) -> Dict[str, Relationship]:
        """Get all relationships from knowledge graph."""
        if self.kg is None:
            return {}
        return self.kg.relationships.copy()

    def get_graph_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        if self.kg is None:
            return {"entities": 0, "relationships": 0}
        return {
            "entities": len(self.kg.entities),
            "relationships": len(self.kg.relationships),
        }

    async def build_from_blocks_lightsag_style(self, blocks: List[ContentBlock], 
                                             llm_config: Optional[LLMConfig] = None, global_config=None) -> None:
        """
        Build knowledge graph from content blocks using LightRAG approach.
        
        Args:
            blocks: List of content blocks
            llm_config: LLMConfig for model selection
            global_config: Global configuration for prompts and settings
        """
        logger.info(f"Building KG from {len(blocks)} blocks using LightRAG approach")

        try:
            # Initialize LLMConfig if not provided
            if llm_config is None:
                llm_config = LLMConfig(
                    model=self.config.llm.model,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    api_key=self.config.llm.api_key,
                    base_url=self.config.llm.base_url
                )
            
            # Initialize LightRAG-style builder
            lightrag_builder = LightRAGKnowledgeGraphBuilder(
                language=self.config.language.default_language,
                llm_config=llm_config,
                global_config=global_config
            )

            # Build entities and relationships using LLM
            await lightrag_builder.build_from_content_blocks(blocks)
            logger.info(f"Extracted {len(lightrag_builder.entities)} entities and {len(lightrag_builder.relationships)} relationships")

            # Store the extracted data for later use
            self._extracted_entities = lightrag_builder.entities
            self._extracted_relationships = lightrag_builder.relationships

        except Exception as e:
            logger.error(f"Failed to build knowledge graph using LightRAG approach: {e}")
            raise

    def rebuild_kuzu_from_extracted_data(self, kuzu_store: KuzuGraphStore) -> None:
        """
        Rebuild Kuzu graph database from LightRAG-style extracted data.
        
        Args:
            kuzu_store: KuzuGraphStore instance
        """
        if not hasattr(self, '_extracted_entities') or not hasattr(self, '_extracted_relationships'):
            raise RuntimeError("No extracted data available. Call build_from_blocks_lightsag_style first.")

        logger.info("Rebuilding Kuzu graph database from extracted data...")
        kuzu_store.rebuild_from_extracted_data(
            self._extracted_entities, 
            self._extracted_relationships
        )

    def rebuild_entities_chroma(self, chroma_kb) -> None:
        """
        (DEPRECATED) Rebuild entities in Chroma knowledge base.
        
        This method is kept for backward compatibility.
        Use embed_entities_and_store_to_chroma() instead.
        
        Args:
            chroma_kb: ChromaKnowledgeBase instance
        """
        logger.warning("rebuild_entities_chroma() is deprecated. Use embed_entities_and_store_to_chroma() instead.")
        
        if not self.kg or not self.entity_embeddings:
            raise RuntimeError("No knowledge graph or entity embeddings available. Call extract_entities_and_relationships first.")

        logger.info("Rebuilding entities in Chroma...")
        chroma_kb.rebuild_entities(self.kg.entities, self.entity_embeddings)

"""
LightRAG-inspired knowledge graph construction and management

This module implements entity and relationship extraction using LLM-driven approach,
following the LightRAG architecture and patterns.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging

from rag_engine.types import ContentBlock, ContentType
from rag_engine.i18n import get_i18n
from rag_engine.config import LLMConfig
from . import prompts
from .llm_client import get_llm_client, LLMClient

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from content"""
    entity_name: str
    entity_type: str
    description: str
    source_id: str = ""
    file_path: str = "unknown_source"
    timestamp: int = field(default_factory=lambda: int(time.time()))


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between entities"""
    src_id: str
    tgt_id: str
    keywords: str
    description: str
    weight: float = 1.0
    source_id: str = ""
    file_path: str = "unknown_source"
    timestamp: int = field(default_factory=lambda: int(time.time()))


class KnowledgeGraph:
    """In-memory knowledge graph storage and management"""
    
    def __init__(self):
        """Initialize empty knowledge graph"""
        self.entities: Dict[str, ExtractedEntity] = {}
        self.relationships: List[ExtractedRelationship] = []
        self.entity_index: Dict[str, List[str]] = {}  # entity_type -> [entity_names]
        
    def add_entity(self, entity: ExtractedEntity):
        """Add an entity to the graph"""
        key = entity.entity_name.lower()
        self.entities[key] = entity
        
        # Update index
        if entity.entity_type not in self.entity_index:
            self.entity_index[entity.entity_type] = []
        if entity.entity_name not in self.entity_index[entity.entity_type]:
            self.entity_index[entity.entity_type].append(entity.entity_name)
    
    def add_relationship(self, relationship: ExtractedRelationship):
        """Add a relationship to the graph"""
        self.relationships.append(relationship)
    
    def get_entity(self, entity_name: str) -> Optional[ExtractedEntity]:
        """Get entity by name"""
        return self.entities.get(entity_name.lower())
    
    def get_entities_by_type(self, entity_type: str) -> List[ExtractedEntity]:
        """Get all entities of a specific type"""
        if entity_type not in self.entity_index:
            return []
        return [self.entities[name.lower()] for name in self.entity_index[entity_type]]
    
    def query_relationships(self, src_id: Optional[str] = None, tgt_id: Optional[str] = None) -> List[ExtractedRelationship]:
        """Query relationships with optional filtering"""
        results = self.relationships
        if src_id:
            results = [r for r in results if r.src_id.lower() == src_id.lower()]
        if tgt_id:
            results = [r for r in results if r.tgt_id.lower() == tgt_id.lower()]
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge graph to dictionary"""
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relationships": [asdict(r) for r in self.relationships]
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'KnowledgeGraph':
        """Create knowledge graph from dictionary"""
        kg = KnowledgeGraph()
        
        # Load entities
        for entity_data in data.get("entities", {}).values():
            entity = ExtractedEntity(**entity_data)
            kg.add_entity(entity)
        
        # Load relationships
        for rel_data in data.get("relationships", []):
            relationship = ExtractedRelationship(**rel_data)
            kg.add_relationship(relationship)
        
        return kg
    
    def save(self, file_path: str):
        """Save knowledge graph to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Knowledge graph saved to: {file_path}")
    
    @staticmethod
    def load(file_path: str) -> 'KnowledgeGraph':
        """Load knowledge graph from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = KnowledgeGraph.from_dict(data)
        logger.info(f"Loaded knowledge graph from: {file_path}")
        return kg


class EntityExtractor:
    """Entity extractor using LLM-based approach (inspired by LightRAG)"""
    
    def __init__(self, language: str = "en", llm_config: Optional[LLMConfig] = None, global_config=None):
        """
        Initialize extractor
        
        Args:
            language: Language for extraction
            llm_config: LLMConfig for model selection (uses global if not provided)
            global_config: Global configuration containing prompts and settings
        """
        self.language = language
        self.global_config = global_config or {}
        self.i18n = get_i18n()
        
        # Initialize LLM client
        self.llm_client = get_llm_client(llm_config)
        
        # Use LightRAG-style system and user prompts
        self.system_prompt = self.global_config.get("entity_extraction_system", 
            prompts.get_system_prompt(language=language))
        
        self.user_prompt_template = self.global_config.get("entity_extraction_user", 
            prompts.ENTITY_EXTRACTION_USER_PROMPT)
        
        # Gleaning/continuation prompts
        self.continue_system_prompt = self.global_config.get("entity_continue_extraction_system",
            prompts.get_continue_extraction_system_prompt())
        
        # Summarization prompts
        self.summarization_system_prompt = self.global_config.get("entity_summarization_system",
            prompts.get_summarization_system_prompt())
        
        self.tuple_delimiter = self.global_config.get("tuple_delimiter", prompts.TUPLE_DELIMITER)
        self.completion_delimiter = self.global_config.get("completion_delimiter", prompts.COMPLETION_DELIMITER)
        
        self.entity_types = self.global_config.get("entity_types", 
            ["Person", "Organization", "Location", "Event", "Product", "Technology", "Concept", "Method", "Data", "Artifact", "Other"])
        
        # Gleaning configuration
        self.max_gleaning_rounds = self.global_config.get("max_gleaning_rounds", 1)
        self.max_summarization_descriptions = self.global_config.get("max_summarization_descriptions", 10)
        self.force_llm_summary = self.global_config.get("force_llm_summary", False)
    
    async def extract_entities_llm(self, content_blocks: List[ContentBlock]) -> List[ExtractedEntity]:
        """
        Extract entities using LLM (LightRAG approach)
        
        Args:
            content_blocks: List of content blocks to extract from
        
        Returns:
            List of extracted entities
        """
        entities = []
        
        for block in content_blocks:
            try:
                block_entities = await self._extract_from_single_block(block)
                entities.extend(block_entities)
            except Exception as e:
                logger.error(f"Failed to extract entities from block {block.id}: {e}")
                continue
        
        return entities
    
    async def extract_entities_with_gleaning(self, content_blocks: List[ContentBlock], 
                                            gleaning_rounds: int = 1) -> List[ExtractedEntity]:
        """
        Extract entities with optional gleaning (refinement) rounds.
        
        Args:
            content_blocks: List of content blocks
            gleaning_rounds: Number of gleaning/refinement rounds (0 = no gleaning)
        
        Returns:
            List of extracted entities with gleaning applied
        """
        # Initial extraction
        entities = await self.extract_entities_llm(content_blocks)
        entity_dict = {e.entity_name.lower(): e for e in entities}
        
        # Apply gleaning rounds if requested
        for round_num in range(min(gleaning_rounds, self.max_gleaning_rounds)):
            logger.info(f"Applying gleaning round {round_num + 1}")
            
            # Extract with gleaning for each block
            for block in content_blocks:
                try:
                    gleaned_entities = await self._gleaning_extraction_from_block(
                        block, entities
                    )
                    
                    # Merge gleaned results
                    for e in gleaned_entities:
                        key = e.entity_name.lower()
                        if key in entity_dict:
                            # Keep version with longer description
                            if len(e.description) > len(entity_dict[key].description):
                                entity_dict[key] = e
                        else:
                            entity_dict[key] = e
                    
                    entities = list(entity_dict.values())
                    
                except Exception as e:
                    logger.warning(f"Gleaning round {round_num + 1} failed for block {block.id}: {e}")
                    continue
        
        return list(entity_dict.values())
    
    async def _extract_from_single_block(self, block: ContentBlock) -> List[ExtractedEntity]:
        """Extract entities from a single content block"""
        content = block.content
        if not content or len(content.strip()) < 10:
            return []
        
        # Get user prompt with actual content
        user_prompt = prompts.get_user_prompt(
            input_text=content,  # Full content for better extraction
            entity_types=self.entity_types,
            tuple_delimiter=self.tuple_delimiter,
            completion_delimiter=self.completion_delimiter,
            language=self.language
        )
        
        try:
            # Call LLM
            response = await self.llm_client.generate_with_system(
                self.system_prompt,
                user_prompt
            )
            
            # Parse response
            entities = self._parse_entity_extraction_response(response, block)
            return entities
            
        except Exception as e:
            logger.error(f"LLM extraction failed for block {block.id}: {e}")
            return []
    
    def _parse_entity_extraction_response(self, response: str, block: ContentBlock) -> List[ExtractedEntity]:
        """Parse LLM response for entity extraction"""
        entities = []
        
        # Split response by lines
        lines = response.strip().split('\n')
        
        # Stop at completion delimiter
        lines_to_process = []
        for line in lines:
            if self.completion_delimiter in line:
                break
            lines_to_process.append(line)
        
        for line in lines_to_process:
            line = line.strip()
            if not line or not line.startswith('entity'):
                continue
            
            try:
                # Parse format: entity<DELIMITER>entity_name<DELIMITER>entity_type<DELIMITER>entity_description
                if self.tuple_delimiter in line:
                    parts = line.split(self.tuple_delimiter)
                    if len(parts) >= 4 and parts[0].strip() == 'entity':
                        entity_name = parts[1].strip()
                        entity_type = parts[2].strip()
                        entity_description = parts[3].strip()
                        
                        if entity_name and entity_type and entity_description:
                            entity = ExtractedEntity(
                                entity_name=entity_name,
                                entity_type=entity_type,
                                description=entity_description,
                                source_id=block.id,
                                file_path=getattr(block, 'source_file', 'unknown_source'),
                                timestamp=int(time.time())
                            )
                            entities.append(entity)
                else:
                    # Fallback: Try regex pattern for <|....|> format
                    pattern = r'entity<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>'
                    match = re.match(pattern, line)
                    
                    if match:
                        entity_name = match.group(1).strip()
                        entity_type = match.group(2).strip()
                        entity_description = match.group(3).strip()
                        
                        if entity_name and entity_type and entity_description:
                            entity = ExtractedEntity(
                                entity_name=entity_name,
                                entity_type=entity_type,
                                description=entity_description,
                                source_id=block.id,
                                file_path=getattr(block, 'source_file', 'unknown_source'),
                                timestamp=int(time.time())
                            )
                            entities.append(entity)
                            
            except Exception as e:
                logger.warning(f"Failed to parse entity line: {line}, error: {e}")
                continue
        
        return entities
    
    async def _gleaning_extraction_from_block(self, block: ContentBlock, 
                                            previously_extracted: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Perform gleaning (refinement) extraction on a block.
        
        Args:
            block: Content block to extract from
            previously_extracted: Previously extracted entities (for context)
        
        Returns:
            List of newly found or corrected entities
        """
        content = block.content
        if not content or len(content.strip()) < 10:
            return []
        
        # Format previous extraction for display
        previous_extraction_text = self._format_entities_for_gleaning(previously_extracted)
        
        # Get gleaning continuation prompt
        continuation_prompt = prompts.get_continue_extraction_prompt(
            input_text=content,  # Full content for gleaning/refinement
            previous_extraction=previous_extraction_text,
            tuple_delimiter=self.tuple_delimiter,
            completion_delimiter=self.completion_delimiter,
            language=self.language
        )
        
        # Combine system + user prompts for gleaning
        full_prompt = f"{self.continue_system_prompt}\n\n{continuation_prompt}"
        
        try:
            response = await self.llm_client.generate_with_system(
                self.continue_system_prompt,
                continuation_prompt
            )
            entities = self._parse_entity_extraction_response(response, block)
            return entities
        except Exception as e:
            logger.error(f"Gleaning extraction failed for block {block.id}: {e}")
            return []
    
    def _format_entities_for_gleaning(self, entities: List[ExtractedEntity]) -> str:
        """Format extracted entities for gleaning prompt."""
        if not entities:
            return "No previous extraction available."
        
        formatted = []
        for e in entities:
            formatted.append(
                f"entity{self.tuple_delimiter}{e.entity_name}{self.tuple_delimiter}{e.entity_type}{self.tuple_delimiter}{e.description}"
            )
        
        return "\n".join(formatted)
    
    async def summarize_descriptions(self, descriptions: List[str], entity_name: str,
                                   description_type: str = "Entity") -> str:
        """
        Summarize multiple descriptions into one using LLM.
        
        Args:
            descriptions: List of descriptions to summarize
            entity_name: Name of entity or relationship
            description_type: "Entity" or "Relationship"
        
        Returns:
            Summarized description
        """
        # If only one description, return it as-is
        if len(descriptions) <= 1:
            return descriptions[0] if descriptions else ""
        
        # If too many descriptions, use map-reduce strategy
        if len(descriptions) > self.max_summarization_descriptions:
            descriptions = await self._map_reduce_summarization(
                descriptions, entity_name, description_type
            )
        
        # If forcing LLM or have multiple descriptions, call LLM
        if self.force_llm_summary or len(descriptions) > 1:
            try:
                prompt = prompts.get_summarization_prompt(
                    descriptions, entity_name, description_type, self.language
                )
                
                summary = await self.llm_client.generate_with_system(
                    self.summarization_system_prompt,
                    prompt
                )
                return summary.strip()
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, using concatenation instead")
        
        # Fallback: concatenate with separator
        return "; ".join(descriptions)
    
    async def _map_reduce_summarization(self, descriptions: List[str], entity_name: str,
                                      description_type: str, max_batch_size: int = 3) -> List[str]:
        """
        Apply map-reduce strategy for summarizing many descriptions.
        
        Args:
            descriptions: List of descriptions
            entity_name: Entity name
            description_type: "Entity" or "Relationship"
            max_batch_size: Batch size for each reduction step
        
        Returns:
            Reduced list of descriptions
        """
        current_batch = descriptions
        
        while len(current_batch) > self.max_summarization_descriptions:
            next_batch = []
            
            # Process in chunks
            for i in range(0, len(current_batch), max_batch_size):
                chunk = current_batch[i:i+max_batch_size]
                
                try:
                    # Summarize this chunk
                    summary = await self.summarize_descriptions(
                        chunk, f"{entity_name} (batch {i//max_batch_size})", description_type
                    )
                    next_batch.append(summary)
                except Exception as e:
                    logger.warning(f"Map-reduce summarization failed: {e}, keeping original chunk")
                    next_batch.extend(chunk)
            
            current_batch = next_batch
        
        return current_batch
    
    async def extract_entities_and_relationships_in_one_call(
        self, content_blocks: List[ContentBlock]
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """
        Extract entities and relationships in a single LLM call (optimized).
        
        This method uses a single LLM request to extract both entities and relationships,
        reducing API calls and costs compared to two separate requests.
        
        Args:
            content_blocks: List of content blocks to extract from
        
        Returns:
            Tuple of (entities list, relationships list)
        """
        entities = []
        relationships = []
        
        logger.info(f"Extracting entities and relationships in single LLM call from {len(content_blocks)} blocks")
        
        for block in content_blocks:
            try:
                block_entities, block_relationships = await self._extract_entities_and_relationships_from_single_block(block)
                entities.extend(block_entities)
                relationships.extend(block_relationships)
            except Exception as e:
                logger.error(f"Failed to extract entities and relationships from block {block.id}: {e}")
                continue
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    async def _extract_entities_and_relationships_from_single_block(
        self, block: ContentBlock
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """
        Extract entities and relationships from a single content block in one LLM call.
        
        Args:
            block: Content block to extract from
        
        Returns:
            Tuple of (entities list, relationships list)
        """
        content = block.content
        if not content or len(content.strip()) < 10:
            return [], []
        
        # Get user prompt with actual content (same system prompt handles both)
        user_prompt = prompts.get_user_prompt(
            input_text=content,  # Limit content length
            entity_types=self.entity_types,
            tuple_delimiter=self.tuple_delimiter,
            completion_delimiter=self.completion_delimiter,
            language=self.language
        )
        
        try:
            # Single LLM call that returns both entities and relationships
            response = await self.llm_client.generate_with_system(
                self.system_prompt,
                user_prompt
            )
            
            # Parse response to extract both entities and relationships
            entities = self._parse_entity_extraction_response(response, block)
            relationships = self._parse_relationship_extraction_response(response, block)
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"LLM extraction failed for block {block.id}: {e}")
            return [], []
    
    def _parse_relationship_extraction_response(
        self, response: str, block: ContentBlock
    ) -> List[ExtractedRelationship]:
        """Parse LLM response to extract relationships."""
        relationships = []
        
        # Split response by lines
        lines = response.strip().split('\n')
        
        # Stop at completion delimiter
        lines_to_process = []
        for line in lines:
            if self.completion_delimiter in line:
                break
            lines_to_process.append(line)
        
        for line in lines_to_process:
            line = line.strip()
            if not line or not line.startswith('relation'):
                continue
            
            try:
                # Parse format: relation<DELIMITER>source_entity<DELIMITER>target_entity<DELIMITER>keywords<DELIMITER>description
                if self.tuple_delimiter in line:
                    parts = line.split(self.tuple_delimiter)
                    if len(parts) >= 5 and parts[0].strip() == 'relation':
                        src_entity = parts[1].strip()
                        tgt_entity = parts[2].strip()
                        keywords = parts[3].strip()
                        description = parts[4].strip()
                        
                        # Infer weight from context (default to 1.0)
                        weight = 1.0
                        if len(parts) > 5:
                            try:
                                weight = float(parts[5].strip())
                            except ValueError:
                                weight = 1.0
                        
                        if src_entity and tgt_entity and description:
                            relationship = ExtractedRelationship(
                                src_id=src_entity,
                                tgt_id=tgt_entity,
                                weight=weight,
                                description=description,
                                keywords=keywords,
                                source_id=block.id,
                                file_path=getattr(block, 'source_file', 'unknown_source'),
                                timestamp=int(time.time())
                            )
                            relationships.append(relationship)
                            
            except Exception as e:
                logger.warning(f"Failed to parse relationship line: {line}, error: {e}")
                continue
        
        return relationships


class RelationshipBuilder:
    """Build relationships using LLM-based approach (inspired by LightRAG)"""
    
    def __init__(self, language: str = "en", llm_config: Optional[LLMConfig] = None, global_config=None):
        """
        Initialize relationship builder
        
        Args:
            language: Language for extraction
            llm_config: LLMConfig for model selection (uses global if not provided)
            global_config: Global configuration
        """
        self.language = language
        self.global_config = global_config or {}
        self.i18n = get_i18n()
        
        # Initialize LLM client
        self.llm_client = get_llm_client(llm_config)
        
        # Use LightRAG-style system prompt for relationships
        self.system_prompt = self.global_config.get("relationship_extraction_system", 
            prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT)
        
        self.tuple_delimiter = self.global_config.get("tuple_delimiter", prompts.TUPLE_DELIMITER)
        self.completion_delimiter = self.global_config.get("completion_delimiter", prompts.COMPLETION_DELIMITER)
        
        # Summarization configuration
        self.max_summarization_descriptions = self.global_config.get("max_summarization_descriptions", 10)
        self.force_llm_summary = self.global_config.get("force_llm_summary", False)
        
        # Summarization prompts
        self.summarization_system_prompt = self.global_config.get("entity_summarization_system",
            prompts.get_summarization_system_prompt())

    def _get_relationship_prompt(self) -> str:
        """Get relationship extraction prompt"""
        return f"""---Task---
Extract relationships between the previously identified entities from the input text.

---Instructions---

1. **Relationship Extraction:** Identify direct, clearly stated relationships between entities.
2. **Format:** Use `relation{self.tuple_delimiter}source_entity{self.tuple_delimiter}target_entity{self.tuple_delimiter}keywords{self.tuple_delimiter}description`
3. **Keywords:** Provide high-level keywords summarizing the relationship, separated by commas (NOT by {self.tuple_delimiter}).
4. **Prioritization:** Output the most significant relationships first.
5. **No Duplicates:** Avoid duplicate relationships with swapped source/target.
6. **Completion:** End with `{self.completion_delimiter}`

---Data to be Processed---
Known Entities: {{entity_list}}

Input Text:
```
{{input_text}}
```

---Output---"""
    
    async def build_relationships_llm(self, content_blocks: List[ContentBlock], entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """
        Build relationships using LLM (LightRAG approach)
        
        Args:
            content_blocks: List of content blocks
            entities: List of extracted entities
        
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        for block in content_blocks:
            block_relationships = await self._extract_relationships_from_block(block, entities)
            relationships.extend(block_relationships)
        
        return relationships
    
    async def _extract_relationships_from_block(self, block: ContentBlock, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships from a single content block using LLM"""
        content = block.content
        if not content or len(content.strip()) < 10:
            return []
        
        # Get entity names for context
        entity_names = list(set(entity.entity_name for entity in entities))
        if len(entity_names) < 2:
            return []
        
        # Prepare prompt
        entity_context = ", ".join(f'"{name}"' for name in entity_names[:30])  # Limit to avoid token overflow
        relationship_prompt = self._get_relationship_prompt().format(
            entity_list=entity_context,
            input_text=content
        )
        
        try:
            # Call LLM
            response = await self.llm_client.generate_with_system(
                self.system_prompt,
                relationship_prompt
            )
            
            # Parse response
            relationships = self._parse_relationship_extraction_response(response, block)
            return relationships
            
        except Exception as e:
            logger.error(f"LLM relationship extraction failed for block {block.id}: {e}")
            return []
    
    def _parse_relationship_extraction_response(self, response: str, block: ContentBlock) -> List[ExtractedRelationship]:
        """Parse LLM response for relationship extraction"""
        relationships = []
        
        # Split response by lines and process each line
        lines = response.strip().split('\n')
        
        # Stop at completion delimiter
        lines_to_process = []
        for line in lines:
            if self.completion_delimiter in line:
                break
            lines_to_process.append(line)
        
        for line in lines_to_process:
            line = line.strip()
            if not line or not line.startswith('relation'):
                continue
            
            try:
                # Parse format: relation<DELIMITER>source_entity<DELIMITER>target_entity<DELIMITER>keywords<DELIMITER>description
                if self.tuple_delimiter in line:
                    parts = line.split(self.tuple_delimiter)
                    if len(parts) >= 5 and parts[0].strip() == 'relation':
                        src_entity = parts[1].strip()
                        tgt_entity = parts[2].strip()
                        keywords = parts[3].strip()
                        description = parts[4].strip()
                        
                        # Infer weight from context (default to 1.0)
                        weight = 1.0
                        if len(parts) > 5:
                            try:
                                weight = float(parts[5].strip())
                            except ValueError:
                                weight = 1.0
                        
                        if src_entity and tgt_entity and description:
                            relationship = ExtractedRelationship(
                                src_id=src_entity,
                                tgt_id=tgt_entity,
                                weight=weight,
                                description=description,
                                keywords=keywords,
                                source_id=block.id,
                                file_path=getattr(block, 'source_file', 'unknown_source'),
                                timestamp=int(time.time())
                            )
                            relationships.append(relationship)
                else:
                    # Fallback: Try regex pattern for <|....|> format
                    pattern = r'relation<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>'
                    match = re.match(pattern, line)
                    
                    if match:
                        src_entity = match.group(1).strip()
                        tgt_entity = match.group(2).strip()
                        keywords = match.group(3).strip()
                        description = match.group(4).strip()
                        weight = 1.0
                        
                        if src_entity and tgt_entity and description:
                            relationship = ExtractedRelationship(
                                src_id=src_entity,
                                tgt_id=tgt_entity,
                                weight=weight,
                                description=description,
                                keywords=keywords,
                                source_id=block.id,
                                file_path=getattr(block, 'source_file', 'unknown_source'),
                                timestamp=int(time.time())
                            )
                            relationships.append(relationship)
                        
            except Exception as e:
                logger.warning(f"Failed to parse relationship line: {line}, error: {e}")
                continue
        
        return relationships


class LightRAGKnowledgeGraphBuilder:
    """Knowledge graph construction using LightRAG approach"""
    
    def __init__(self, language: str = "en", llm_config: Optional[LLMConfig] = None, global_config=None):
        """
        Initialize LightRAG-style knowledge graph builder
        
        Args:
            language: Language for processing
            llm_config: LLMConfig for model selection (uses global if not provided)
            global_config: Global configuration
        """
        self.language = language
        self.global_config = global_config or {}
        
        self.entity_extractor = EntityExtractor(language, llm_config, global_config)
        self.relationship_builder = RelationshipBuilder(language, llm_config, global_config)
        
        # Storage for extracted data
        self.entities: Dict[str, ExtractedEntity] = {}
        self.relationships: List[ExtractedRelationship] = []
        
        logger.info("LightRAGKnowledgeGraphBuilder initialized with LLM-based extraction")
    
    async def build_from_content_blocks(self, content_blocks: List[ContentBlock]) -> Tuple[Dict[str, ExtractedEntity], List[ExtractedRelationship]]:
        """
        Build knowledge graph from content blocks using LightRAG approach
        
        Args:
            content_blocks: List of content blocks
        
        Returns:
            Tuple of (entities_dict, relationships_list)
        """
        logger.info(f"Building knowledge graph from {len(content_blocks)} content blocks using LightRAG approach")
        logger.info(f"Using LLM model: {self.entity_extractor.llm_client.config.model}")
        
        # Extract entities using LLM (with optional gleaning)
        gleaning_rounds = self.global_config.get("max_gleaning_rounds", 0)
        if gleaning_rounds > 0:
            extracted_entities = await self.entity_extractor.extract_entities_with_gleaning(
                content_blocks, gleaning_rounds
            )
        else:
            extracted_entities = await self.entity_extractor.extract_entities_llm(content_blocks)
        
        # Merge entities with summarization (LightRAG style)
        self.entities = await self._merge_entities_with_summarization(extracted_entities)
        
        # Extract relationships using LLM
        extracted_relationships = await self.relationship_builder.build_relationships_llm(
            content_blocks, list(self.entities.values())
        )
        
        # Merge relationships with summarization
        self.relationships = await self._merge_relationships_with_summarization(extracted_relationships)
        
        logger.info(f"Knowledge graph built: {len(self.entities)} entities, {len(self.relationships)} relationships")
        
        return self.entities, self.relationships
    
    async def _merge_entities_with_summarization(self, extracted_entities: List[ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """
        Merge extracted entities with description summarization (LightRAG style).
        
        Args:
            extracted_entities: List of extracted entities
        
        Returns:
            Merged entities dictionary with summarized descriptions
        """
        merged = {}
        
        # First pass: group entities by name
        for entity in extracted_entities:
            key = entity.entity_name.lower()
            
            if key in merged:
                # Store multiple descriptions for later summarization
                if not hasattr(merged[key], '_descriptions'):
                    merged[key]._descriptions = [merged[key].description]
                merged[key]._descriptions.append(entity.description)
                
                # Update timestamp if newer
                if entity.timestamp > merged[key].timestamp:
                    merged[key].timestamp = entity.timestamp
            else:
                entity._descriptions = [entity.description]
                merged[key] = entity
        
        # Second pass: summarize descriptions
        for key in merged:
            entity = merged[key]
            if hasattr(entity, '_descriptions') and len(entity._descriptions) > 1:
                try:
                    # Summarize multiple descriptions
                    summarized = await self.entity_extractor.summarize_descriptions(
                        entity._descriptions, entity.entity_name, "Entity"
                    )
                    entity.description = summarized
                except Exception as e:
                    logger.warning(f"Failed to summarize entity {entity.entity_name}: {e}")
                    # Keep concatenation as fallback
                    entity.description = "; ".join(entity._descriptions)
            
            # Clean up temporary attribute
            if hasattr(entity, '_descriptions'):
                delattr(entity, '_descriptions')
        
        return merged
    
    async def _merge_relationships_with_summarization(self, extracted_relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """
        Merge extracted relationships with keyword aggregation and description summarization.
        
        Args:
            extracted_relationships: List of extracted relationships
        
        Returns:
            Merged relationships list
        """
        merged = {}
        
        # First pass: aggregate by source->target
        for rel in extracted_relationships:
            key = f"{rel.src_id.lower()}->{rel.tgt_id.lower()}"
            
            if key in merged:
                existing = merged[key]
                
                # Aggregate weights (take maximum to indicate strongest connection)
                existing.weight = max(existing.weight, rel.weight)
                
                # Store multiple descriptions for later summarization
                if not hasattr(existing, '_descriptions'):
                    existing._descriptions = [existing.description]
                existing._descriptions.append(rel.description)
                
                # Merge keywords
                existing.keywords = self._merge_keywords(existing.keywords, rel.keywords)
                
                # Update timestamp if newer
                if rel.timestamp > existing.timestamp:
                    existing.timestamp = rel.timestamp
            else:
                rel._descriptions = [rel.description]
                merged[key] = rel
        
        # Second pass: summarize descriptions
        for key in merged:
            rel = merged[key]
            if hasattr(rel, '_descriptions') and len(rel._descriptions) > 1:
                try:
                    # Summarize multiple descriptions
                    summarized = await self.entity_extractor.summarize_descriptions(
                        rel._descriptions, f"{rel.src_id} -> {rel.tgt_id}", "Relationship"
                    )
                    rel.description = summarized
                except Exception as e:
                    logger.warning(f"Failed to summarize relationship {rel.src_id}->{rel.tgt_id}: {e}")
                    # Keep concatenation as fallback
                    rel.description = "; ".join(rel._descriptions)
            
            # Clean up temporary attribute
            if hasattr(rel, '_descriptions'):
                delattr(rel, '_descriptions')
        
        return list(merged.values())
    
    def _merge_entities(self, extracted_entities: List[ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """
        Simple entity merging (without summarization).
        For backward compatibility. Use _merge_entities_with_summarization for full LightRAG behavior.
        
        Args:
            extracted_entities: List of extracted entities
        
        Returns:
            Merged entities dictionary
        """
        merged = {}
        
        for entity in extracted_entities:
            key = entity.entity_name.lower()
            
            if key in merged:
                # Merge descriptions
                existing = merged[key]
                existing.description = self._merge_descriptions(existing.description, entity.description)
                # Update timestamp if newer
                if entity.timestamp > existing.timestamp:
                    existing.timestamp = entity.timestamp
            else:
                merged[key] = entity
        
        return merged
    
    def _merge_relationships(self, extracted_relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """
        Merge extracted relationships (LightRAG style)
        
        Args:
            extracted_relationships: List of extracted relationships
        
        Returns:
            Merged relationships list
        """
        merged = {}
        
        for rel in extracted_relationships:
            key = f"{rel.src_id.lower()}->{rel.tgt_id.lower()}"
            
            if key in merged:
                # Merge weights and descriptions
                existing = merged[key]
                existing.weight = max(existing.weight, rel.weight)
                existing.description = self._merge_descriptions(existing.description, rel.description)
                existing.keywords = self._merge_keywords(existing.keywords, rel.keywords)
            else:
                merged[key] = rel
        
        return list(merged.values())
    
    def _merge_descriptions(self, desc1: str, desc2: str) -> str:
        """Merge two descriptions"""
        if not desc1:
            return desc2
        if not desc2:
            return desc1
        
        # Simple concatenation with separator
        return f"{desc1}; {desc2}"
    
    def _merge_keywords(self, kw1: str, kw2: str) -> str:
        """Merge keywords"""
        if not kw1:
            return kw2
        if not kw2:
            return kw1
        
        # Combine unique keywords
        keywords1 = set(kw1.split(','))
        keywords2 = set(kw2.split(','))
        combined = keywords1.union(keywords2)
        
        return ', '.join(combined)
    
    def get_entities(self) -> Dict[str, ExtractedEntity]:
        """Get all entities"""
        return self.entities
    
    def get_relationships(self) -> List[ExtractedRelationship]:
        """Get all relationships"""
        return self.relationships

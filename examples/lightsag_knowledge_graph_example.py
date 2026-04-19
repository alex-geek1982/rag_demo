"""
LightRAG-Style Knowledge Graph Construction Example

This example demonstrates how to build a knowledge graph using LLM-driven 
entity and relationship extraction based on the LightRAG approach.

Features:
1. LLM-powered entity extraction (instead of simple block-to-entity mapping)
2. LLM-powered relationship extraction (instead of sequential relationships)
3. Support for entity and relationship merging and deduplication
4. References LightRAG's data structures and processing logic
5. Uses professional English prompts from LightRAG's actual implementation

Running with PowerShell:
$env:DEEPBRICKS_API_KEY = "<your-key>"
python examples/lightsag_knowledge_graph_example.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine.config import (
    EmbeddingConfig,
    LLMConfig,
    PDFProcessingConfig,
    ProcessingConfig,
    RAGEngineConfig,
    VisionConfig,
    LanguageConfig,
)
from rag_engine.core import create_engine
from rag_engine.pipeline import DocumentProcessor
from rag_engine.storage import ChromaKnowledgeBase, KuzuGraphStore
from rag_engine.retrieval import OpenAIEmbedding


async def llm_extraction_function(prompt: str) -> str:
    """
    Mock LLM function for entity and relationship extraction.
    In production, this should call a real LLM API.
    """
    # Return mock responses based on prompt content
    if "extract relationships" in prompt.lower() or ("known entities" in prompt.lower() and "relation" in prompt.lower()):
        # Mock relationship extraction response
        return """relation<|#|>Order System<|#|>Scheduling Algorithm<|#|>utilizes, orchestrates<|#|>The Order System utilizes the Scheduling Algorithm to intelligently distribute and allocate orders based on inventory and delivery capabilities.
relation<|#|>User<|#|>Order System<|#|>interacts with, depends on<|#|>Users interact with the Order System to submit purchase requests and track order status.
relation<|#|>Order System<|#|>Microservice Architecture<|#|>implements, based on<|#|>The Order System is implemented using Microservice Architecture to enable scalability and independent deployment.
relation<|#|>Scheduling Algorithm<|#|>Microservice Architecture<|#|>deployed in, part of<|#|>The Scheduling Algorithm is deployed as a microservice within the overall Microservice Architecture.
relation<|#|>API Gateway<|#|>Microservice Architecture<|#|>interfaces with, coordinates<|#|>The API Gateway serves as the interface between external clients and the Microservice Architecture.
<|COMPLETE|>"""
    elif "extract" in prompt.lower() and "entit" in prompt.lower():
        # Mock entity extraction response using LightRAG format with <|#|> delimiter
        return """entity<|#|>Order System<|#|>Technology<|#|>The order system is a critical component of e-commerce platforms, responsible for processing customer orders and managing order fulfillment workflows.
entity<|#|>Scheduling Algorithm<|#|>Technology<|#|>The scheduling algorithm is an advanced computational method designed to optimize order allocation and distribution logistics.
entity<|#|>User<|#|>Person<|#|>Users are customers and end-users of the e-commerce platform who interact with the order system to make purchases.
entity<|#|>Microservice Architecture<|#|>Technology<|#|>Microservice architecture is a cloud-native design pattern that decomposes the system into independent, loosely-coupled services.
entity<|#|>API Gateway<|#|>Technology<|#|>The API Gateway is a central component that manages external communications and request routing for all microservices.
<|COMPLETE|>"""
    else:
        return ""


def get_global_config() -> Dict[str, Any]:
    """Get global configuration using LightRAG-style English prompts"""
    from rag_engine.core import prompts
    
    return {
        # Use LightRAG-style system and user prompts (English)
        "entity_extraction_system": prompts.get_system_prompt(language="English"),
        "entity_extraction_user": prompts.ENTITY_EXTRACTION_USER_PROMPT,
        
        # Delimiters
        "tuple_delimiter": prompts.TUPLE_DELIMITER,
        "completion_delimiter": prompts.COMPLETION_DELIMITER,
        
        # Entity types
        "entity_types": [
            "Person", "Organization", "Location", "Event", 
            "Product", "Technology", "Concept", "Method", "Data", "Artifact", "Other"
        ]
    }


async def main():
    """Main function"""
    print("=== LightRAG-Style Knowledge Graph Construction Example ===\n")
    
    # Sample content blocks (in English)
    sample_content_blocks = [
        {
            "id": "block_1",
            "content": "The Order System is a critical component of e-commerce platforms, responsible for processing customer orders and managing fulfillment workflows. The system uses advanced scheduling algorithms to optimize order allocation and ensure efficient logistics delivery.",
            "type": "text",
            "page_num": 1,
            "file_path": "product_spec.pdf"
        },
        {
            "id": "block_2", 
            "content": "Users can submit purchase requests through the Order System, which intelligently schedules deliveries based on inventory and delivery capacity. The scheduling algorithm considers multiple dimensions including time, cost, and customer satisfaction.",
            "type": "text",
            "page_num": 2,
            "file_path": "product_spec.pdf"
        },
        {
            "id": "block_3",
            "content": "The technical architecture employs microservices design, with the Order System, User Service, and Scheduling Service deployed independently. Communication occurs through an API Gateway, ensuring high availability and scalability of the system.",
            "type": "text",
            "page_num": 3,
            "file_path": "product_spec.pdf"
        }
    ]
    
    # Convert to ContentBlock objects
    from rag_engine.types import ContentBlock, ContentType, ModalityType
    content_blocks = []
    for block_data in sample_content_blocks:
        block = ContentBlock(
            id=block_data["id"],
            content=block_data["content"],
            type=ContentType(block_data["type"]),
            modality=ModalityType.TEXT,  # Default text modality
            source_file=block_data["file_path"],
            page_num=block_data["page_num"],
            language="en"
        )
        content_blocks.append(block)
    
    print(f"Processing {len(content_blocks)} content blocks with LightRAG-style extraction\n")
    
    # Get global configuration
    global_config = get_global_config()
    
    try:
        # Use LightRAG-style builder directly
        print("=== Starting LightRAG-Style Knowledge Graph Construction ===")
        from rag_engine.core.knowledge_graph import LightRAGKnowledgeGraphBuilder
        
        lightrag_builder = LightRAGKnowledgeGraphBuilder(
            language="en",
            llm_func=llm_extraction_function,
            global_config=global_config
        )
        
        # Build entities and relationships
        entities, relationships = await lightrag_builder.build_from_content_blocks(content_blocks)
        
        print("✅ Knowledge graph construction completed")
        
        # Display construction results
        print(f"\n📊 Construction Results:")
        print(f"   Entities extracted: {len(entities)}")
        print(f"   Relationships extracted: {len(relationships)}")
        
        print(f"\n🏷️  Extracted Entities:")
        for entity_name, entity in entities.items():
            print(f"   - {entity_name} ({entity.entity_type}): {entity.description}")
        
        print(f"\n🔗 Extracted Relationships:")
        for rel in relationships:
            print(f"   - {rel.src_id} --[{rel.keywords}]--> {rel.tgt_id} (weight: {rel.weight})")
            print(f"     Description: {rel.description}")
        
        print("\n=== Example Completed ===")
        
    except Exception as e:
        print(f"❌ Error during construction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())
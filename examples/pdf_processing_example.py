"""
Example: Advanced PDF processing with layout understanding

Demonstrates how to:
1. Extract images and surrounding text from PDFs
2. Generate image descriptions using Vision API
3. Process tables and structured data
4. Create multimodal documents
"""

import logging
from pathlib import Path
from rag_engine.config import RAGEngineConfig, PDFProcessingConfig
from rag_engine.parsers import PDFParser
from rag_engine import RAGEngine
from rag_engine.types import ContentType


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_pdf_processing():
    """Example 1: Basic PDF processing with advanced layout understanding"""
    logger.info("=" * 60)
    logger.info("Example 1: Basic PDF Processing with Layout Understanding")
    logger.info("=" * 60)
    
    # Create PDF parser with advanced layout processing
    parser = PDFParser(
        doc_id="paper_001",
        doc_title="Research Paper with Images",
        language="en",
        use_advanced_layout=True,  # Enable advanced PDF processing
        extract_images=True,
        extract_tables=True,
        use_vision_api=True
    )
    
    # Example PDF path (you need to provide a real PDF)
    pdf_path = "example.pdf"
    
    # Only process if file exists
    if Path(pdf_path).exists():
        try:
            document = parser.parse(pdf_path)
            
            logger.info(f"\n✅ Document parsed successfully!")
            logger.info(f"   Total content blocks: {len(document.content_blocks)}")
            
            # Count content types
            text_blocks = [b for b in document.content_blocks if b.type == ContentType.TEXT]
            image_blocks = [b for b in document.content_blocks if b.type == ContentType.IMAGE]
            table_blocks = [b for b in document.content_blocks if b.metadata.get("block_type") == "table"]
            
            logger.info(f"   📄 Text blocks: {len(text_blocks)}")
            logger.info(f"   🖼️  Image blocks: {len(image_blocks)}")
            logger.info(f"   📊 Table blocks: {len(table_blocks)}")
            
            # Show first few text blocks
            logger.info("\n📄 Text Blocks:")
            for i, block in enumerate(text_blocks[:2]):
                preview = block.content[:80].replace('\n', ' ')
                logger.info(f"   Block {i}: {preview}...")
            
            # Show images with descriptions
            logger.info("\n🖼️  Images with Descriptions:")
            for i, block in enumerate(image_blocks[:3]):
                description = block.metadata.get("description", "No description")
                position = block.metadata.get("position", {})
                logger.info(f"   Image {i}: {block.content}")
                logger.info(f"     Page: {block.page_num}")
                logger.info(f"     Position: ({position.get('x0'):.0f}, {position.get('y0'):.0f}) "
                           f"to ({position.get('x1'):.0f}, {position.get('y1'):.0f})")
                logger.info(f"     Description: {description[:100]}...")
                
                # Show surrounding context
                context = block.metadata.get("surrounding_text", "")
                if context:
                    logger.info(f"     Context: {context[:80]}...")
            
            # Show tables
            logger.info("\n📊 Tables:")
            for i, block in enumerate(table_blocks[:2]):
                logger.info(f"   Table {i}:")
                logger.info(f"   {block.content[:200]}...")
        
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
    else:
        logger.info(f"PDF file not found: {pdf_path}")
        logger.info("Please provide a valid PDF file to process.")


def example_2_custom_pdf_config():
    """Example 2: Using custom PDF processing configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Custom PDF Processing Configuration")
    logger.info("=" * 60)
    
    # Create custom PDF processing config
    pdf_config = PDFProcessingConfig(
        use_advanced_layout=True,
        extract_images=True,
        extract_tables=True,
        use_vision_api=True,
        context_window_pixels=300,  # Increase context search range
        min_image_area=500,  # Lower minimum image area
        max_surrounding_text_chars=3000  # Increase context limit
    )
    
    # Create RAG engine config with custom PDF settings
    rag_config = RAGEngineConfig(
        pdf_processing=pdf_config,
        debug=False
    )
    
    logger.info("\n📋 PDF Processing Configuration:")
    logger.info(f"   Advanced Layout: {pdf_config.use_advanced_layout}")
    logger.info(f"   Extract Images: {pdf_config.extract_images}")
    logger.info(f"   Extract Tables: {pdf_config.extract_tables}")
    logger.info(f"   Use Vision API: {pdf_config.use_vision_api}")
    logger.info(f"   Context Window: {pdf_config.context_window_pixels} pixels")
    logger.info(f"   Min Image Area: {pdf_config.min_image_area} pixels")
    logger.info(f"   Max Context Chars: {pdf_config.max_surrounding_text_chars}")


def example_3_rag_engine_integration():
    """Example 3: Using RAG Engine with advanced PDF processing"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: RAG Engine Integration")
    logger.info("=" * 60)
    
    # Initialize RAG engine (automatically uses advanced PDF processing)
    config = RAGEngineConfig.from_env()
    engine = RAGEngine(config)
    
    pdf_path = "research_paper.pdf"
    
    if Path(pdf_path).exists():
        try:
            # Process PDF document
            document = engine.process_document(
                file_path=pdf_path,
                doc_id="research_001",
                doc_title="Research Paper"
            )
            
            logger.info(f"\n✅ Document processed by RAG Engine!")
            logger.info(f"   Document: {document.title}")
            logger.info(f"   Content blocks: {len(document.content_blocks)}")
            
            # Analyze multimodal content
            for block in document.content_blocks[:10]:
                block_type = block.type.value
                modality = block.modality.value
                
                if block_type == "image":
                    size = block.metadata.get("position", {})
                    logger.info(f"   🖼️  Image (page {block.page_num}): "
                              f"{size.get('width', 0):.0f}x{size.get('height', 0):.0f}px")
                else:
                    chars = len(block.content)
                    logger.info(f"   📄 Text (page {block.page_num}): {chars} chars")
        
        except Exception as e:
            logger.error(f"Failed: {e}")
    else:
        logger.info(f"PDF not found: {pdf_path}")


def example_4_analyze_images_with_context():
    """Example 4: Analyze extracted images with their context"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Analyze Images with Context")
    logger.info("=" * 60)
    
    parser = PDFParser(
        doc_id="analysis_001",
        doc_title="Document for Analysis",
        use_advanced_layout=True,
        extract_images=True
    )
    
    pdf_path = "document_with_figures.pdf"
    
    if Path(pdf_path).exists():
        try:
            document = parser.parse(pdf_path)
            
            # Extract and analyze images
            image_blocks = [b for b in document.content_blocks if b.type == ContentType.IMAGE]
            
            logger.info(f"\n📊 Found {len(image_blocks)} images in document\n")
            
            for idx, img_block in enumerate(image_blocks, 1):
                logger.info(f"Image {idx}:")
                logger.info(f"  Path: {img_block.content}")
                logger.info(f"  Page: {img_block.page_num}")
                
                # Image description
                description = img_block.metadata.get("description", "No description")
                logger.info(f"  📝 Description: {description[:150]}...")
                
                # Related text context
                context = img_block.metadata.get("surrounding_text", "")
                if context:
                    logger.info(f"  💬 Related Text: {context[:150]}...")
                
                # Position information
                position = img_block.metadata.get("position", {})
                logger.info(f"  📍 Position: Page {img_block.page_num}, "
                          f"({position.get('x0', 0):.0f}, {position.get('y0', 0):.0f}) "
                          f"Size: {position.get('width', 0):.0f}x{position.get('height', 0):.0f}px")
                
                # Related text block count
                related_count = img_block.metadata.get("related_text_block_count", 0)
                logger.info(f"  📚 Related text blocks: {related_count}\n")
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
    else:
        logger.info(f"PDF not found: {pdf_path}")


def example_5_batch_pdf_processing():
    """Example 5: Batch process multiple PDFs"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Batch PDF Processing")
    logger.info("=" * 60)
    
    pdf_dir = Path("./pdfs")
    
    if not pdf_dir.exists():
        logger.info(f"PDF directory not found: {pdf_dir}")
        logger.info("Create a './pdfs' directory with PDF files to test batch processing.")
        return
    
    config = RAGEngineConfig.from_env()
    engine = RAGEngine(config)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"\nFound {len(pdf_files)} PDF files\n")
    
    for pdf_file in pdf_files[:5]:  # Process first 5
        try:
            logger.info(f"Processing: {pdf_file.name}")
            document = engine.process_document(
                file_path=str(pdf_file),
                doc_id=f"pdf_{pdf_file.stem}",
                doc_title=pdf_file.stem
            )
            
            # Statistics
            total_blocks = len(document.content_blocks)
            text_blocks = sum(1 for b in document.content_blocks if b.type == ContentType.TEXT)
            image_blocks = sum(1 for b in document.content_blocks if b.type == ContentType.IMAGE)
            
            logger.info(f"  ✅ Processed successfully")
            logger.info(f"     - Total blocks: {total_blocks}")
            logger.info(f"     - Text blocks: {text_blocks}")
            logger.info(f"     - Image blocks: {image_blocks}\n")
        
        except Exception as e:
            logger.error(f"  ❌ Failed: {e}\n")


def example_6_performance_comparison():
    """Example 6: Compare performance of different configurations"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Performance Configuration Comparison")
    logger.info("=" * 60)
    
    # Configuration 1: High accuracy (slower)
    high_accuracy_config = PDFProcessingConfig(
        use_advanced_layout=True,
        extract_images=True,
        extract_tables=True,
        use_vision_api=True,
        context_window_pixels=400,
        max_surrounding_text_chars=5000
    )
    
    # Configuration 2: Balanced
    balanced_config = PDFProcessingConfig(
        use_advanced_layout=True,
        extract_images=True,
        extract_tables=True,
        use_vision_api=True,
        context_window_pixels=200,
        max_surrounding_text_chars=2000
    )
    
    # Configuration 3: Fast (lower quality)
    fast_config = PDFProcessingConfig(
        use_advanced_layout=True,
        extract_images=False,  # Skip image extraction
        extract_tables=False,  # Skip table extraction
        use_vision_api=False,  # No Vision API
        context_window_pixels=100
    )
    
    configs = {
        "High Accuracy": high_accuracy_config,
        "Balanced": balanced_config,
        "Fast": fast_config
    }
    
    logger.info("\n📊 Configuration Comparison:\n")
    logger.info(f"{'Config':<15} {'Adv Layout':<12} {'Images':<10} {'Tables':<10} {'Vision':<10} {'Speed':<10}")
    logger.info("-" * 65)
    
    for name, config in configs.items():
        speed = "⚡⚡⚡" if name == "Fast" else "⚡⚡" if name == "Balanced" else "⚡"
        logger.info(f"{name:<15} {str(config.use_advanced_layout):<12} "
                   f"{str(config.extract_images):<10} {str(config.extract_tables):<10} "
                   f"{str(config.use_vision_api):<10} {speed:<10}")


def main():
    """Run all examples"""
    logger.info("\n🚀 Advanced PDF Processing Examples\n")
    
    # Run examples
    example_2_custom_pdf_config()
    example_3_rag_engine_integration()
    example_6_performance_comparison()
    
    # Optional examples (require actual PDF files)
    # example_1_basic_pdf_processing()
    # example_4_analyze_images_with_context()
    # example_5_batch_pdf_processing()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Examples completed!")
    logger.info("=" * 60)
    logger.info("\nNote: Examples 1, 4, 5 require actual PDF files.")
    logger.info("See PDF_LAYOUT_UNDERSTANDING.md for more details.")


if __name__ == "__main__":
    main()

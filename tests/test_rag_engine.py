"""
Tests for RAG Engine
"""
import unittest
from pathlib import Path
import tempfile
import os

from rag_engine.types import ContentType, ModalityType, ContentBlock, Document
from rag_engine.parsers import TextParser, ParserFactory
from rag_engine.parsers.pdf_advanced import AdvancedPDFProcessor, TextBlock, ImageLocation
from rag_engine.processors import TextProcessor, ProcessorFactory
from rag_engine.pipeline.document_processor import DocumentProcessor
from rag_engine.config import RAGEngineConfig
from rag_engine.i18n import I18n, get_i18n


class TestContentTypes(unittest.TestCase):
    """Test content type enums"""
    
    def test_content_types(self):
        """Test ContentType enum"""
        self.assertEqual(ContentType.TEXT.value, "text")
        self.assertEqual(ContentType.IMAGE.value, "image")
        self.assertEqual(ContentType.TABLE.value, "table")
    
    def test_modality_types(self):
        """Test ModalityType enum"""
        self.assertEqual(ModalityType.TEXT.value, "text")
        self.assertEqual(ModalityType.VISUAL.value, "visual")


class TestContentBlock(unittest.TestCase):
    """Test ContentBlock class"""
    
    def test_content_block_creation(self):
        """Test creating content block"""
        block = ContentBlock(
            id="block_1",
            type=ContentType.TEXT,
            content="Test content",
            modality=ModalityType.TEXT
        )
        
        self.assertEqual(block.id, "block_1")
        self.assertEqual(block.content, "Test content")
        self.assertEqual(block.type, ContentType.TEXT)
    
    def test_content_block_to_dict(self):
        """Test converting to dict"""
        block = ContentBlock(
            id="block_1",
            type=ContentType.TEXT,
            content="Test",
            modality=ModalityType.TEXT
        )
        
        data = block.to_dict()
        self.assertEqual(data["id"], "block_1")
        self.assertEqual(data["type"], "text")


class TestTextParser(unittest.TestCase):
    """Test text file parser"""
    
    def test_text_parser_support(self):
        """Test parser support check"""
        parser = TextParser("doc1", "Test Doc")
        
        self.assertTrue(parser.supports("file.txt"))
        self.assertTrue(parser.supports("file.md"))
        self.assertFalse(parser.supports("file.pdf"))
    
    def test_text_parser_parse(self):
        """Test parsing text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            temp_path = f.name
        
        try:
            parser = TextParser("doc1", "Test Doc")
            doc = parser.parse(temp_path)
            
            self.assertEqual(doc.id, "doc1")
            self.assertEqual(len(doc.content_blocks), 1)
            self.assertIn("test", doc.content_blocks[0].content.lower())
        finally:
            os.unlink(temp_path)


class TestParserFactory(unittest.TestCase):
    """Test parser factory"""
    
    def test_get_text_parser(self):
        """Test getting text parser"""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            parser = ParserFactory.get_parser(f.name, "doc1", "Test")
            self.assertIsInstance(parser, TextParser)


class TestTextProcessor(unittest.TestCase):
    """Test text content processor"""
    
    def test_text_processor_support(self):
        """Test content type support"""
        processor = TextProcessor()
        
        self.assertTrue(processor.supports(ContentType.TEXT))
        self.assertFalse(processor.supports(ContentType.IMAGE))
    
    def test_text_processor_process(self):
        """Test processing text block"""
        block = ContentBlock(
            id="block_1",
            type=ContentType.TEXT,
            content="This is a test sentence. This is another sentence.",
            modality=ModalityType.TEXT
        )
        
        processor = TextProcessor()
        description, entity = processor.process(block)
        
        self.assertIsNotNone(description)
        self.assertIsNotNone(entity)
        self.assertEqual(entity.type, "TextContent")


class TestProcessorFactory(unittest.TestCase):
    """Test processor factory"""
    
    def test_get_text_processor(self):
        """Test getting text processor"""
        processor = ProcessorFactory.get_processor(ContentType.TEXT)
        self.assertIsInstance(processor, TextProcessor)
    
    def test_get_image_processor(self):
        """Test getting image processor"""
        from rag_engine.processors import ImageProcessor
        processor = ProcessorFactory.get_processor(ContentType.IMAGE)
        self.assertIsInstance(processor, ImageProcessor)


class TestDocumentProcessorMarkdownOrder(unittest.TestCase):
    """Test markdown conversion order for mixed content blocks"""

    def test_image_ocr_description_preserved_in_position(self):
        """Image OCR descriptions should appear adjacent to the image in markdown."""
        processor = DocumentProcessor(RAGEngineConfig())
        document = Document(
            id="doc1",
            title="Test Doc",
            source_path="test.pdf",
            language="en",
            content_blocks=[
                ContentBlock(
                    id="text1",
                    type=ContentType.TEXT,
                    content="Introduction text.",
                    modality=ModalityType.TEXT,
                    page_num=0,
                    metadata={"position": {"x0": 0, "y0": 10, "x1": 100, "y1": 20}},
                ),
                ContentBlock(
                    id="image1",
                    type=ContentType.IMAGE,
                    content="image1.png",
                    modality=ModalityType.VISUAL,
                    page_num=0,
                    metadata={
                        "description": "This is image 1.",
                        "position": {"x0": 0, "y0": 30, "x1": 100, "y1": 40},
                    },
                ),
                ContentBlock(
                    id="text2",
                    type=ContentType.TEXT,
                    content="Conclusion text.",
                    modality=ModalityType.TEXT,
                    page_num=0,
                    metadata={"position": {"x0": 0, "y0": 50, "x1": 100, "y1": 60}},
                ),
            ],
        )

        markdown = processor._convert_to_markdown(document)
        self.assertIn("Introduction text.", markdown)
        self.assertIn("![This is image 1.](image1.png)", markdown)
        self.assertIn("Conclusion text.", markdown)
        self.assertLess(markdown.index("Introduction text."), markdown.index("![This is image 1.](image1.png)"))
        self.assertLess(markdown.index("![This is image 1.](image1.png)"), markdown.index("Conclusion text."))

    def test_bullet_point_character_normalization(self):
        """Special PDF bullet characters should normalize to a visible bullet."""
        processor = DocumentProcessor(RAGEngineConfig())
        document = Document(
            id="doc2",
            title="Bullet Test",
            source_path="test.pdf",
            language="en",
            content_blocks=[
                ContentBlock(
                    id="text1",
                    type=ContentType.TEXT,
                    content="\uf06c Item one",
                    modality=ModalityType.TEXT,
                    page_num=0,
                    metadata={"position": {"x0": 0, "y0": 10, "x1": 100, "y1": 20}},
                ),
            ],
        )

        markdown = processor._convert_to_markdown(document)
        self.assertIn("• Item one", markdown)
        self.assertNotIn("\\uf06c", markdown)
        self.assertNotIn("\uf06c", markdown)


class TestAdvancedPDFProcessor(unittest.TestCase):
    """Test advanced PDF processing helpers"""

    def test_filter_header_footer_blocks_removes_repeated_noise(self):
        """Repeated headers and page-number footers should be dropped"""
        processor = AdvancedPDFProcessor(
            use_vision_api=False,
            header_margin_ratio=0.1,
            footer_margin_ratio=0.1,
        )
        blocks = [
            TextBlock("Confidential Report", 0, 0, 10, 200, 30, "text"),
            TextBlock("Main body on page one", 0, 20, 120, 320, 180, "text"),
            TextBlock("Page 1", 0, 20, 760, 120, 790, "text"),
            TextBlock("Confidential Report", 1, 0, 10, 200, 30, "text"),
            TextBlock("Main body on page two", 1, 20, 140, 320, 200, "text"),
            TextBlock("Page 2", 1, 20, 760, 120, 790, "text"),
        ]

        filtered = processor._filter_header_footer_blocks(blocks, page_heights={0: 800, 1: 800})
        filtered_texts = [block.text for block in filtered]

        self.assertIn("Main body on page one", filtered_texts)
        self.assertIn("Main body on page two", filtered_texts)
        self.assertNotIn("Confidential Report", filtered_texts)
        self.assertNotIn("Page 1", filtered_texts)
        self.assertNotIn("Page 2", filtered_texts)

    def test_filter_header_footer_blocks_keeps_real_top_heading(self):
        """A real document heading near the top should stay if it is not repeated"""
        processor = AdvancedPDFProcessor(
            use_vision_api=False,
            header_margin_ratio=0.1,
            footer_margin_ratio=0.1,
        )
        blocks = [
            TextBlock("Executive Summary", 0, 30, 35, 260, 70, "text"),
            TextBlock("The quarterly results improved significantly.", 0, 30, 110, 380, 170, "text"),
        ]

        filtered = processor._filter_header_footer_blocks(blocks, page_heights={0: 800})
        filtered_texts = [block.text for block in filtered]

        self.assertIn("Executive Summary", filtered_texts)
        self.assertIn("The quarterly results improved significantly.", filtered_texts)

    def test_filter_header_footer_blocks_keeps_repeated_page_heading(self):
        """Repeated page headings should stay unless they are clearly header/footer content"""
        processor = AdvancedPDFProcessor(
            use_vision_api=False,
            header_margin_ratio=0.1,
            footer_margin_ratio=0.1,
        )
        blocks = [
            TextBlock("版本修订", 0, 30, 35, 260, 70, "text"),
            TextBlock("章节内容第一段。", 0, 30, 110, 380, 170, "text"),
            TextBlock("版本修订", 1, 30, 35, 260, 70, "text"),
            TextBlock("章节内容第二段。", 1, 30, 110, 380, 170, "text"),
            TextBlock("Page 1", 0, 20, 760, 120, 790, "text"),
            TextBlock("Page 2", 1, 20, 760, 120, 790, "text"),
        ]

        filtered = processor._filter_header_footer_blocks(blocks, page_heights={0: 800, 1: 800})
        filtered_texts = [block.text for block in filtered]

        self.assertIn("版本修订", filtered_texts)
        self.assertIn("章节内容第一段。", filtered_texts)
        self.assertIn("章节内容第二段。", filtered_texts)
        self.assertNotIn("Page 1", filtered_texts)
        self.assertNotIn("Page 2", filtered_texts)

    def test_filter_header_footer_images_removes_repeated_logo(self):
        """Repeated margin logo images should be excluded from RAG content"""
        processor = AdvancedPDFProcessor(
            use_vision_api=False,
            header_margin_ratio=0.1,
            footer_margin_ratio=0.1,
        )
        images = [
            ImageLocation("logo_p1.png", 0, 467.4, 26.8, 522.96, 47.2, 55.6, 20.4),
            ImageLocation("logo_p2.png", 1, 467.4, 26.8, 522.96, 47.2, 55.6, 20.4),
            ImageLocation("figure.png", 1, 80.0, 260.0, 420.0, 520.0, 340.0, 260.0),
        ]

        filtered = processor._filter_header_footer_images(images, page_heights={0: 800, 1: 800})
        filtered_paths = [img.image_path for img in filtered]

        self.assertEqual(filtered_paths, ["figure.png"])

    def test_extract_images_converts_bottom_origin_y_to_top_coordinates(self):
        """Image y0/y1 from bottom-origin page.images should convert to top-origin coordinates."""
        class DummyPage:
            height = 1000
            images = [{"x0": 100.0, "x1": 300.0, "y0": 200.0, "y1": 500.0}]

        processor = AdvancedPDFProcessor(use_vision_api=False)
        processor._extract_and_save_image = lambda page, img_obj, page_num, pdf_path, doc_id, img_idx: "dummy.png"

        images = processor._extract_images_from_page(DummyPage(), 0, "dummy.pdf", "doc1")

        self.assertEqual(len(images), 1)
        image_loc = images[0]
        self.assertEqual(image_loc.y0, 500.0)
        self.assertEqual(image_loc.y1, 800.0)

    def test_get_surrounding_text_splits_above_and_below_correctly(self):
        """Text blocks above and below the image should be grouped into correct context fields."""
        processor = AdvancedPDFProcessor(use_vision_api=False)
        image_loc = ImageLocation("dummy.png", 0, 100.0, 400.0, 200.0, 500.0, 100.0, 100.0)
        text_blocks = [
            TextBlock("2. 权限管理", 0, 0.0, 0.0, 500.0, 50.0, "text"),
            TextBlock("简要说明：", 0, 0.0, 50.0, 500.0, 120.0, "text"),
            TextBlock("界面原型：", 0, 0.0, 240.0, 500.0, 280.0, "text"),
            TextBlock("图2-1 岗位查询界面", 0, 0.0, 520.0, 500.0, 560.0, "text"),
        ]

        surrounding = processor._get_surrounding_text(image_loc, text_blocks)
        self.assertIn("界面原型：", surrounding["context_above"])
        self.assertEqual(surrounding["context_below"], "图2-1 岗位查询界面")

    def test_get_surrounding_text_ignores_page_markers(self):
        """Page marker text blocks should not be included in surrounding text."""
        processor = AdvancedPDFProcessor(use_vision_api=False)
        image_loc = ImageLocation("dummy.png", 0, 100.0, 400.0, 200.0, 500.0, 100.0, 100.0)
        text_blocks = [
            TextBlock("Some caption here", 0, 0.0, 520.0, 500.0, 560.0, "text"),
            TextBlock("5", 0, 0.0, 580.0, 500.0, 600.0, "text"),
        ]

        surrounding = processor._get_surrounding_text(image_loc, text_blocks)
        self.assertEqual(surrounding["context_below"], "Some caption here")


class TestI18n(unittest.TestCase):
    """Test internationalization"""
    
    def test_i18n_default_language(self):
        """Test default language"""
        i18n = I18n("en")
        self.assertEqual(i18n.get_language(), "en")
    
    def test_i18n_translation(self):
        """Test getting translations"""
        i18n = get_i18n("en")
        text = i18n.t("label_text")
        self.assertIsNotNone(text)
    
    def test_i18n_language_switch(self):
        """Test switching languages"""
        i18n = I18n("en")
        
        i18n.set_language("zh")
        self.assertEqual(i18n.get_language(), "zh")
        
        i18n.set_language("en")
        self.assertEqual(i18n.get_language(), "en")
    
    def test_i18n_supported_languages(self):
        """Test getting supported languages"""
        i18n = I18n("en")
        langs = i18n.get_supported_languages()
        self.assertIn("en", langs)
        self.assertIn("zh", langs)


if __name__ == "__main__":
    unittest.main()

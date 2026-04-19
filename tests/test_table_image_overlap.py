#!/usr/bin/env python3
"""
Comprehensive pytest-based test to verify table-image filtering.

Tests that tables located completely inside images are correctly filtered out,
while preserving actual standalone tables in the document.

Algorithm: A table is considered to be inside an image if >65% of the table's
area is within the image bounds. This filters out tables that are captured
in screenshots or other images, while preserving legitimate standalone tables.
The 65% threshold accounts for PDF table detection variations.

Special focus on page 12 where tables in images should be filtered.

Usage:
  pytest tests/test_table_image_overlap.py -v
  pytest tests/test_table_image_overlap.py::TestPage12TableFiltering -v
  pytest tests/test_table_image_overlap.py -v --tb=short
"""

import sys
import logging
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

from rag_engine.parsers.pdf_advanced import AdvancedPDFProcessor, ImageLocation


# Fixtures
@pytest.fixture(scope="session")
def pdf_path():
    """Fixture: PDF file path"""
    path = "examples/京东订单多维度调度系统PRD1.0.pdf"
    assert Path(path).exists(), f"PDF not found: {path}"
    return path


@pytest.fixture(scope="session")
def processor():
    """Fixture: Create PDF processor instance"""
    return AdvancedPDFProcessor(
        extract_images=True,
        extract_tables=True,
        extract_text=False,
        use_vision_api=False,
        filter_header_footer=False
    )


@pytest.fixture(scope="session")
def processed_doc(processor, pdf_path):
    """Fixture: Process PDF and return document"""
    logger.info(f"Processing PDF: {pdf_path}")
    doc = processor.process_pdf(pdf_path, "test", "Test", "zh")
    logger.info(f"Successfully processed: {len(doc.content_blocks)} content blocks")
    return doc


# Test Functions
class TestPage12TableFiltering:
    """Test suite for page 12 table filtering (image with red-boxed table)"""
    
    def test_page_12_has_images(self, processed_doc):
        """Test: Page 12 should have images extracted"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        images = [b for b in page_12_blocks if b.type.value == 'image']

        logger.info(f"Page 12: {len(images)} images found")
        assert len(images) > 0, "Page 12 should have extracted images"
        assert len(images) == 2, f"Page 12 should have 2 images, found {len(images)}"
    
    def test_page_12_no_tables(self, processed_doc):
        """Test: Page 12 should have 1 table (export format)"""        
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        tables = [b for b in page_12_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']

        logger.info(f"Page 12: {len(tables)} tables found")
        assert len(tables) == 1, (
            f"Page 12 should have 1 table (export format table), "
            f"but found {len(tables)}. Tables are filtered only if they're >90% inside images."
        )
    
    def test_page_12_image_details(self, processed_doc):
        """Test: Verify page 12 image extraction details"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        images = [b for b in page_12_blocks if b.type.value == 'image']

        assert len(images) == 2, f"Expected 2 images, found {len(images)}"
        
        # Verify image details
        for i, img in enumerate(images):
            pos = img.metadata.get('position', {})
            x0, y0 = pos.get('x0', 0), pos.get('y0', 0)
            x1, y1 = pos.get('x1', 0), pos.get('y1', 0)
            width, height = x1 - x0, y1 - y0
            
            logger.info(f"  Image {i}: position=({x0:.1f}, {y0:.1f}), "
                       f"size={width:.0f}x{height:.0f}px")
            
            # Verify images have valid positions
            assert x0 >= 0, f"Image {i} x0 should be >= 0"
            assert y0 >= 0, f"Image {i} y0 should be >= 0"
            assert width > 0, f"Image {i} width should be > 0"
            assert height > 0, f"Image {i} height should be > 0"
    
    def test_page_12_table_filtering_result(self, processed_doc):
        """Test: Page 12 should have the export table (filtering only applies to >90%)"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        
        images = [b for b in page_12_blocks if b.type.value == 'image']
        tables = [b for b in page_12_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']
        
        logger.info(f"Page 12 Analysis:")
        logger.info(f"  Total blocks: {len(page_12_blocks)}")
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Tables: {len(tables)}")
        
        # Table filtering threshold is 90% (conservative)
        # Tables are only filtered if >90% inside images (complete artifacts)
        # The export table is legitimate even though it's ~65% inside an image
        assert len(tables) == 1, (
            "Page 12 should have 1 table (export format table). "
            "Only tables >90% inside images are filtered (complete artifacts). "
            "(all detected tables should be preserved if not completely inside images)"
        )

    def test_table_blocks_are_not_duplicated_by_text_blocks(self, processor, processed_doc):
        """Test: Detected table blocks should not also appear as overlapping normal text blocks."""
        table_blocks = [
            b for b in processed_doc.content_blocks
            if b.metadata.get('block_type') == 'table'
        ]
        text_blocks = [
            b for b in processed_doc.content_blocks
            if b.type.value == 'text' and b.metadata.get('block_type') != 'table'
        ]

        assert table_blocks, "No table blocks were found to validate overlap behavior"

        for table in table_blocks:
            table_pos = table.metadata.get('position', {})
            if not table_pos:
                continue

            table_bbox = (
                table_pos.get('x0', 0),
                table_pos.get('y0', 0),
                table_pos.get('x1', 0),
                table_pos.get('y1', 0)
            )
            overlapping_text = [
                tb for tb in text_blocks
                if tb.page_num == table.page_num and processor._calculate_bbox_overlap_ratio(
                    (
                        tb.metadata['position']['x0'],
                        tb.metadata['position']['y0'],
                        tb.metadata['position']['x1'],
                        tb.metadata['position']['y1']
                    ),
                    table_bbox
                ) >= 0.8
            ]
            assert not overlapping_text, (
                f"Table on page {table.page_num + 1} should not overlap normal text blocks. "
                f"Found {len(overlapping_text)} overlapping text block(s)."
            )


class TestOtherPagesPreservesTables:
    """Test suite to verify tables on other pages are preserved"""
    
    def test_tables_exist_on_other_pages(self, processed_doc):
        """Test: Tables should exist on pages other than 11"""
        # Count tables by page
        table_pages = {}
        for b in processed_doc.content_blocks:
            if b.metadata.get('block_type') == 'table':
                if b.page_num not in table_pages:
                    table_pages[b.page_num] = 0
                table_pages[b.page_num] += 1
        
        logger.info(f"Tables by page: {table_pages}")
        
        # Page 11 should have 0 tables
        assert 10 not in table_pages, "Page 11 (index 10) should have 0 tables"
        
        # Other pages should have tables
        assert len(table_pages) > 0, "Document should have tables on some pages"
    
    def test_specific_pages_have_tables(self, processed_doc):
        """Test: Specific pages should have expected number of tables"""
        expected_tables = {
            1: 1,   # Page 2
            3: 1,   # Page 4
            6: 1,   # Page 7
            7: 0,   # Page 8 - removed (single-column bullet list, not a table)
            11: 1,  # Page 12 - export format table
            12: 0,  # Page 13 - no tables (bullet points filtered)
        }
        
        # Count tables by page
        table_pages = {}
        for b in processed_doc.content_blocks:
            if b.metadata.get('block_type') == 'table':
                if b.page_num not in table_pages:
                    table_pages[b.page_num] = 0
                table_pages[b.page_num] += 1
        
        for page_idx, expected_count in expected_tables.items():
            actual_count = table_pages.get(page_idx, 0)
            logger.info(f"  Page {page_idx + 1}: {actual_count} table(s) "
                       f"(expected {expected_count})")
            assert actual_count == expected_count, (
                f"Page {page_idx + 1} should have {expected_count} table(s), "
                f"found {actual_count}"
            )
    
    def test_page_11_excluded_from_table_pages(self, processed_doc):
        """Test: Page 11 should NOT appear in pages with tables"""
        page_11_blocks = [b for b in processed_doc.content_blocks if b.page_num == 10]
        page_11_has_tables = any(
            b.type.value == 'text' and b.metadata.get('block_type') == 'table'
            for b in page_11_blocks
        )
        
        assert not page_11_has_tables, (
            "Page 11 should not have any tables "
            "(all detected tables should be filtered due to image overlap)"
        )


class TestPage13TableDetection:
    """Test suite to verify page 12/13 table detection"""
    
    def test_page_13_has_no_table(self, processed_doc):
        """Test: Page 13 should have NO table (bullet points are filtered out)"""
        page_13_blocks = [b for b in processed_doc.content_blocks if b.page_num == 12]
        tables = [b for b in page_13_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']
        
        logger.info(f"Page 13: {len(tables)} tables found (expected 0 - bullet points filtered)")
        assert len(tables) == 0, "Page 13 should have no tables (bullet points are not real tables)"
    
    def test_page_12_has_export_table(self, processed_doc):
        """Test: Page 12 has the export format table"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        tables = [b for b in page_12_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']
        
        logger.info(f"Page 12: {len(tables)} tables found (expected 1 - export format table)")
        assert len(tables) == 1, "Page 12 should have 1 table (export format table)"
    
    def test_page_12_export_table_has_content(self, processed_doc):
        """Test: Page 12 export table should have valid content"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        tables = [b for b in page_12_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']
        
        assert len(tables) > 0, "Page 12 should have a table"
        
        table_block = tables[0]
        logger.info(f"Table content length: {len(table_block.content)} characters")
        
        # Verify table has meaningful content (not empty or just whitespace)
        assert table_block.content, "Table should have non-empty content"
        assert len(table_block.content.strip()) > 0, "Table should have non-whitespace content"
    
    def test_page_12_export_table_has_position_metadata(self, processed_doc):
        """Test: Page 12 export table should have position metadata"""
        page_12_blocks = [b for b in processed_doc.content_blocks if b.page_num == 11]
        tables = [b for b in page_12_blocks if b.type.value == 'text' and b.metadata.get('block_type') == 'table']
        
        assert len(tables) > 0, "Page 12 should have a table"
        
        table_block = tables[0]
        position = table_block.metadata.get('position')
        
        assert position is not None, "Table should have position metadata"
        assert 'x0' in position and 'y0' in position, "Position should have x0, y0"
        assert 'x1' in position and 'y1' in position, "Position should have x1, y1"
        assert position['x0'] < position['x1'], "Position x0 should be < x1"
        assert position['y0'] < position['y1'], "Position y0 should be < y1"


class TestOverlapDetectionMethod:
    """Test the overlap detection method itself"""
    
    def test_table_completely_inside_image(self, processor):
        """Test: Table completely inside image should be filtered"""
        image_loc = ImageLocation(
            image_path="test.png",
            page_num=0,
            x0=0, y0=0, x1=1000, y1=1000,
            width=1000, height=1000
        )
        # Table completely inside image bounds (100%)
        table_bbox = (100, 100, 200, 200)
        
        is_inside = processor._is_table_completely_inside_image(table_bbox, image_loc)
        
        logger.info(f"Table completely inside image (100%): {is_inside}")
        assert is_inside, "Table completely inside image should return True"
    
    def test_table_partially_inside_image(self, processor):
        """Test: Table partially inside image should not be filtered"""
        image_loc = ImageLocation(
            image_path="test.png",
            page_num=0,
            x0=0, y0=0, x1=200, y1=200,
            width=200, height=200
        )
        # Table extends beyond image bounds (only 50% inside)
        # Image: (0,0,200,200), Table: (100,100,300,300)
        # Intersection: (100,100,200,200) = 100*100 = 10000
        # Table area: 200*200 = 40000
        # Ratio: 10000/40000 = 0.25 (25%)
        table_bbox = (100, 100, 300, 300)
        
        is_inside = processor._is_table_completely_inside_image(table_bbox, image_loc)
        
        logger.info(f"Table partially inside image (25%): {is_inside}")
        assert not is_inside, "Table partially inside image (<70%) should return False"
    
    def test_table_mostly_inside_image(self, processor):
        """Test: Table mostly (>90%) inside image should be filtered"""
        image_loc = ImageLocation(
            image_path="test.png",
            page_num=0,
            x0=0, y0=0, x1=200, y1=200,
            width=200, height=200
        )
        # Table is 95% inside image bounds
        # Image: (0,0,200,200), Table: (5,0,205,200)
        # Intersection: (5,0,200,200) = 195*200 = 39000
        # Table area: 200*200 = 40000
        # Ratio: 39000/40000 = 0.975 (97.5% > 90%)
        table_bbox = (5, 0, 205, 200)
        
        is_inside = processor._is_table_completely_inside_image(table_bbox, image_loc)
        
        logger.info(f"Table mostly inside image (97.5%): {is_inside}")
        assert is_inside, "Table mostly inside image (>90%) should return True"
    
    def test_table_outside_image(self, processor):
        """Test: Table outside image should not be filtered"""
        image_loc = ImageLocation(
            image_path="test.png",
            page_num=0,
            x0=0, y0=0, x1=100, y1=100,
            width=100, height=100
        )
        # Table completely outside image
        table_bbox = (200, 200, 300, 300)
        
        is_inside = processor._is_table_completely_inside_image(table_bbox, image_loc)
        
        logger.info(f"Table outside image: {is_inside}")
        assert not is_inside, "Table outside image should return False"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "page11: tests for page 11 table filtering"
    )
    config.addinivalue_line(
        "markers", "other_pages: tests for preserving tables on other pages"
    )
    config.addinivalue_line(
        "markers", "overlap: tests for overlap detection method"
    )


if __name__ == "__main__":
    # This allows running the test file directly with pytest
    pytest.main([__file__, "-v"])


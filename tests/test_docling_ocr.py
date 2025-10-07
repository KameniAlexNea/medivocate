"""Tests for Docling OCR processor."""
from unittest.mock import MagicMock, patch

from ocr.config import OCRConfig
from ocr.docling_processor import DoclingOCRProcessor


class TestDoclingOCRProcessor:
    """Test Docling OCR processor functionality."""

    @patch("ocr.docling_processor.DocumentConverter")
    def test_initialization(self, mock_converter):
        """Test DoclingOCRProcessor initialization."""
        config = OCRConfig(engine='docling')
        processor = DoclingOCRProcessor(config)

        assert processor.config == config
        mock_converter.assert_called_once()

    @patch('ocr.docling_processor.DocumentConverter')
    def test_process_pdf_markdown(self, mock_converter):
        """Test PDF processing with markdown output and page breaks."""
        # Mock the converter and result
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance

        mock_result = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        # Simulate markdown with page breaks
        mock_result.document.export_to_markdown.return_value = "# Page 1\n\nContent 1\n---PAGE_BREAK---\n# Page 2\n\nContent 2"

        config = OCRConfig(engine='docling')
        processor = DoclingOCRProcessor(config)

        results = processor.process_pdf('test.pdf')

        assert len(results) == 2
        assert results[0] == (0, "# Page 1\n\nContent 1")
        assert results[1] == (1, "# Page 2\n\nContent 2")
        mock_converter_instance.convert.assert_called_once_with('test.pdf')

    @patch('ocr.docling_processor.DocumentConverter')
    def test_process_image(self, mock_converter):
        """Test image processing."""
        # Mock the converter and result
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance

        mock_result = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_result.document.export_to_markdown.return_value = "# Image Content"

        config = OCRConfig(engine='docling')
        processor = DoclingOCRProcessor(config)

        result = processor.process_image('test.jpg')

        assert result == "# Image Content"
        mock_converter_instance.convert.assert_called_once_with('test.jpg')

    @patch('ocr.docling_processor.DocumentConverter')
    def test_process_pdf_markdown(self, mock_converter):
        """Test PDF processing with markdown output and page breaks."""
        # Mock the converter and result
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance

        mock_result = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        # Simulate markdown with page breaks
        mock_result.document.export_to_markdown.return_value = "# Page 1\n\nContent 1\n---PAGE_BREAK---\n# Page 2\n\nContent 2"

        config = OCRConfig(engine='docling')
        processor = DoclingOCRProcessor(config)

        results = processor.process_pdf('test.pdf')

        assert len(results) == 2
        assert results[0] == (0, "# Page 1\n\nContent 1")
        assert results[1] == (1, "# Page 2\n\nContent 2")
        mock_converter_instance.convert.assert_called_once_with('test.pdf')

    @patch('ocr.docling_processor.DocumentConverter')
    def test_process_image(self, mock_converter):
        """Test image processing."""
        # Mock the converter and result
        mock_converter_instance = MagicMock()
        mock_converter.return_value = mock_converter_instance

        mock_result = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_result.document.export_to_markdown.return_value = "# Image Content"

        config = OCRConfig(engine='docling')
        processor = DoclingOCRProcessor(config)

        result = processor.process_image('test.jpg')

        assert result == "# Image Content"
        mock_converter_instance.convert.assert_called_once_with('test.jpg')

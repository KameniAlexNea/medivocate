from pathlib import Path
from typing import List, Optional

from docling.document_converter import DocumentConverter

from .config import OCRConfig


class DoclingOCRProcessor:
    """OCR processor using Docling for advanced document processing."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.converter = DocumentConverter()

    def process_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> List[tuple]:
        """Process PDF file and extract text from pages using Docling.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to process (0-indexed), None for all

        Returns:
            List of tuples (page_number, text)
        """
        # Convert document using Docling
        result = self.converter.convert(pdf_path)

        # Export based on configured format
        if self.config.docling_format == "markdown":
            full_text = result.document.export_to_markdown()
        elif self.config.docling_format == "json":
            full_text = result.document.export_to_dict()
            # Convert dict to string representation
            import json
            full_text = json.dumps(full_text, indent=2, ensure_ascii=False)
        else:  # text
            full_text = result.document.export_to_text()

        # For now, return as single page since Docling processes the whole document
        # In the future, we could split by pages if needed
        return [(0, full_text)]

    def process_image(self, image_path: str) -> str:
        """Process single image file using Docling.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        # Docling can handle images directly
        result = self.converter.convert(image_path)

        if self.config.docling_format == "markdown":
            return result.document.export_to_markdown()
        elif self.config.docling_format == "json":
            import json
            return json.dumps(result.document.export_to_dict(), indent=2, ensure_ascii=False)
        else:  # text
            return result.document.export_to_text()
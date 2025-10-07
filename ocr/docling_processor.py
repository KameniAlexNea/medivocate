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

        # Export to markdown with page break placeholders
        page_break_placeholder = "\n---PAGE_BREAK---\n"
        full_markdown = result.document.export_to_markdown(
            page_break_placeholder=page_break_placeholder
        )

        # Split by page break placeholder to get individual pages
        page_texts = full_markdown.split(page_break_placeholder)

        # Return list of (page_number, text) tuples
        results = []
        for page_num, page_text in enumerate(page_texts):
            if page_text.strip():  # Only include non-empty pages
                results.append((page_num, page_text.strip()))

        return results

    def process_image(self, image_path: str) -> str:
        """Process single image file using Docling.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        # Docling can handle images directly
        result = self.converter.convert(image_path)

        # Always export to markdown for consistency
        return result.document.export_to_markdown()

from typing import List, Optional

import pymupdf

from .config import OCRConfig


class PDFProcessor:
    """Processor for machine-readable PDF documents using PyMuPDF."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()

    def process_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> List[tuple]:
        """Process PDF file and extract text directly from pages.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to process (0-indexed), None for all

        Returns:
            List of tuples (page_number, text)
        """
        doc = pymupdf.open(pdf_path)

        if pages is None:
            pages = list(range(len(doc)))

        results = []

        for page_num in pages:
            page = doc[page_num]
            text = page.get_text()

            # Only include pages with actual text content
            if text.strip():
                results.append((page_num, text.strip()))

        doc.close()
        return results

    def process_image(self, image_path: str) -> str:
        """Process single image file.

        Note: This processor is designed for PDFs. For images, it returns empty string.
        Use OCR processors for image text extraction.

        Args:
            image_path: Path to image file

        Returns:
            Empty string (not applicable for PDF processor)
        """
        return ""

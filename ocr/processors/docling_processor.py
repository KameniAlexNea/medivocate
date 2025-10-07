from typing import List, Optional

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from .config import OCRConfig


class DoclingOCRProcessor:
    """OCR processor using Docling for advanced document processing."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_VLLM,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )

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

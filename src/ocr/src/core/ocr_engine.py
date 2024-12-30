import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import easyocr
from tqdm import tqdm

from ..config.ocr_config import OCRConfig
from ..core.image_handler import ImageHandler
from ..core.pdf_handler import PDFHandler, PDFPage
from ..enums.ocr_enum import OutputFormat

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR results and metadata.

    Attributes:
        text: Extracted text
        confidence: Confidence score
        bounding_box: Coordinates of text location
        page_number: Page number (for PDF)
        language: Detected language
    """

    text: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    page_number: Optional[int] = None
    language: Optional[str] = None


class OCREngine:
    """Main OCR processing engine.

    Coordinates between PDF/Image handlers and EasyOCR to perform text extraction
    with support for multiple languages, batch processing, and various output formats.

    Attributes:
        pdf_handler: PDFHandler instance
        image_handler: ImageHandler instance
        config: OCR configuration
        reader: EasyOCR reader instance
    """

    def __init__(
        self, pdf_handler: PDFHandler, image_handler: ImageHandler, config: OCRConfig
    ):
        """Initialize OCR engine with handlers and configuration.

        Args:
            pdf_handler: Instance of PDFHandler
            image_handler: Instance of ImageHandler
            config: OCR configuration
        """
        self.pdf_handler = pdf_handler
        self.image_handler = image_handler
        self.config = config
        self.reader = easyocr.Reader(
            config.languages, gpu=True if self._check_gpu() else False
        )

    def process_file(
        self, file_path: Union[str, Path], output_format: Optional[OutputFormat] = None
    ) -> Union[Dict, str]:
        """Process a file (PDF or image) and extract text.

        Args:
            file_path: Path to file
            output_format: Desired output format (defaults to config setting)

        Returns:
            Extracted text in specified format

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.suffix.lower() == ".pdf":
                results = self._process_pdf(file_path)
            else:
                results = self._process_image(file_path)

            return self._format_output(
                results, output_format or self.config.output_format
            )

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def _process_pdf(self, pdf_path: Path) -> List[List[OCRResult]]:
        """Process PDF document and extract text from all pages.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of OCR results per page
        """
        pages = self.pdf_handler.pdf_to_images(
            pdf_path, self.config.dpi, batch_size=self.config.batch_size
        )

        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            results = list(tqdm(executor.map(self._process_single_page, pages), total=len(pages)))

        return results

    def _process_single_page(self, page: PDFPage) -> List[OCRResult]:
        """Process a single page and extract text.

        Args:
            page: PDFPage instance

        Returns:
            List of OCR results for the page
        """
        processed = self.image_handler.preprocess_image(
            page.image, self.config.preprocessing
        )

        raw_results = self.reader.readtext(processed.image)
        return [
            OCRResult(
                text=text,
                confidence=conf,
                bounding_box=bbox,
                page_number=page.page_number,
            )
            for bbox, text, conf in raw_results
        ]

    def _process_image(self, image_path: Path) -> List[OCRResult]:
        """Process single image file.

        Args:
            image_path: Path to image file

        Returns:
            List of OCR results
        """
        image = self.image_handler.load_image(image_path)
        processed = self.image_handler.preprocess_image(
            image, self.config.preprocessing
        )

        raw_results = self.reader.readtext(processed.image)
        return [
            OCRResult(text=text, confidence=conf, bounding_box=bbox)
            for bbox, text, conf in raw_results
        ]

    @staticmethod
    def _check_gpu() -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _to_json(self, results: List[OCRResult]) -> str:
        import json

        output = {
            "pages": [
                {
                    "page_number": 1,  # Single page assumption
                    "content": [
                        {
                            "text": result.text,
                            "confidence": result.confidence,
                            # "bounding_box": result.bounding_box
                        }
                        # Directly iterate through flat list
                        for result in results
                    ],
                }
            ]
        }
        return json.dumps(output, indent=2)

    def _to_xml(self, results: List) -> str:
        root = ET.Element("document")
        for i, page in enumerate(results):
            page_elem = ET.SubElement(root, "page", number=str(i + 1))
            for text in page:
                text_elem = ET.SubElement(page_elem, "text")
                text_elem.text = text[1]
                text_elem.set("confidence", str(text[2]))
                bbox = text[0]
                text_elem.set("bbox", f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
        return ET.tostring(root, encoding="unicode", method="xml")

    def _to_text(self, results: List[OCRResult]) -> str:
        """Convert OCR results to plain text format.

        Organizes text by page and preserves layout using relative positioning.

        Args:
            results: List of OCR results

        Returns:
            Formatted text string with page separators
        """
        # Handle empty input
        if not results:
            return "No OCR results available."

        # Group results by page
        pages = self._group_by_page(results)

        # Generate text output
        output = []
        for page_num in sorted(pages):  # Iterate over pages in ascending order
            output.append(self._format_page_header(page_num))
            output.append(self._format_page_content(pages[page_num]))

        return "".join(output)

    def _group_by_page(self, results: List[OCRResult]) -> dict:
        """Group OCR results by page number."""
        pages = {}
        for result in results:
            page_num = result.page_number
            pages.setdefault(page_num, []).append(result)
        return pages

    def _format_page_header(self, page_num: int) -> str:
        """Format header for each page."""
        header = f"\n{'=' * 40}\nPage {page_num}\n{'=' * 40}\n"
        return header

    def _format_page_content(self, results: List[OCRResult]) -> str:
        """Format the content of a single page, preserving layout."""
        # Sort results by vertical position (top-to-bottom)
        results = sorted(results, key=lambda x: self._get_y_coord(x))

        output, line_buffer = [], []
        current_line_y = None
        line_tolerance = 10  # Y-coordinate tolerance to detect line breaks

        for result in results:
            y_coord = self._get_y_coord(result)  # Get the Y-coordinate

            # Check if we're starting a new line
            if current_line_y is None or abs(y_coord - current_line_y) > line_tolerance:
                # Append buffered line and reset
                if line_buffer:
                    output.append(" ".join(line_buffer))
                    line_buffer = []  # Reset line buffer
                current_line_y = y_coord

            # Add the text to the current line buffer
            line_buffer.append(result.text)

        # Flush any remaining buffered line
        if line_buffer:
            output.append(" ".join(line_buffer))

        return "\n".join(output) + "\n"

    def _get_y_coord(self, result: OCRResult) -> float:
        """Extract Y-coordinate from the bounding box."""
        # The bounding box is a list of 4 coordinate pairs [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # The Y-coordinate is the second element of the first pair: bounding_box[0][1]
        try:
            # Extract the Y-coordinate as a float
            return float(result.bounding_box[0][1])
        except (IndexError, TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid bounding box data for result: {result}. Error: {e}"
            )

    def _format_output(
        self, results: List[OCRResult], format_type: OutputFormat
    ) -> Union[Dict, str]:
        """Format OCR results according to specified output format.

        Args:
            results: List of OCR results
            format_type: Desired output format

        Returns:
            Formatted results in specified format
        """
        if format_type == OutputFormat.JSON:
            return self._to_json(results)
        elif format_type == OutputFormat.XML:
            return self._to_xml(results)
        else:
            return self._to_text(results)

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import easyocr

from ocr.config.ocr_config import OCRConfig
from ocr.core.image_handler import ImageHandler
from ocr.enums.ocr_enum import OutputFormat
from ocr.core.pdf_base_handler import PDFHandler

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    page_number: Optional[int] = None
    language: Optional[str] = None


class BaseEngine:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.image_handler = ImageHandler()
        self.pdf_handler = PDFHandler()
        self.reader = easyocr.Reader(
            config.languages, gpu=True if self._check_gpu() else False
        )

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _process_image(self, image_path: Path, image_page: int) -> List[OCRResult]:
        image = self.image_handler.load_image(image_path)
        processed = self.image_handler.preprocess_image(
            image, self.config.preprocessing
        )

        raw_results = self.reader.readtext(processed.image)
        return [
            OCRResult(
                text=text, confidence=conf, bounding_box=bbox, page_number=image_page
            )
            for bbox, text, conf in raw_results
        ]

    def _process_images(
        self, image_paths: list[Path], image_pages: list[int]
    ) -> List[OCRResult]:
        images = [
            self.image_handler.load_image(image_path) for image_path in image_paths
        ]
        processed_images = [
            self.image_handler.preprocess_image(image, self.config.preprocessing)
            for image in images
        ]

        raw_results_batch = self.reader.readtext_batched(
            [processed.image for processed in processed_images]
        )
        return [
            [
                OCRResult(
                    text=text,
                    confidence=conf,
                    bounding_box=bbox,
                    page_number=image_page,
                )
                for bbox, text, conf in raw_results
            ]
            for raw_results, image_page in zip(raw_results_batch, image_pages)
        ]


class OCREngine(BaseEngine):
    def __init__(self, config):
        super().__init__(config)

    def _to_json(self, results: List[OCRResult]) -> str:
        output = {
            "pages": [
                {
                    "page_number": 1,  # Single page assumption
                    "content": [
                        {
                            "text": result.text,
                            "confidence": result.confidence,
                            "bounding_box": result.bounding_box,
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
        # Handle empty input
        if not results:
            return ""

        # Group results by page
        pages = self._group_by_page(results)

        # Generate text output
        output = []
        for page_num in sorted(pages):  # Iterate over pages in ascending order
            output.append(self._format_page_content(pages[page_num]))

        return "".join(output)

    def _group_by_page(self, results: List[OCRResult]) -> dict:
        """Group OCR results by page number."""
        pages = {}
        for result in results:
            page_num = result.page_number
            pages.setdefault(page_num, []).append(result)
        return pages

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
        if format_type == OutputFormat.JSON:
            return self._to_json(results)
        elif format_type == OutputFormat.XML:
            return self._to_xml(results)
        else:
            return self._to_text(results)

    def convert_pdf_to_data(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ):
        for batch_images in self.pdf_handler.pdf_to_images_batch(
            pdf_path, self.config.dpi, pages, self.config.batch_size
        ):
            img_pages = [batch.page_number for batch in batch_images]
            batch_processed = self._process_images(
                [batch.image for batch in batch_images], img_pages
            )
            yield [bs.page_number for bs in batch_images], [
                self._format_output(page, self.config.output_format)
                for page in batch_processed
            ]

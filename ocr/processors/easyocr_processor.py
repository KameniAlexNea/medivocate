from typing import List, Optional

import easyocr
import numpy as np
from PIL import Image

from .config import OCRConfig
from ..utils import pdf_to_images, preprocess_image


class EasyOCRProcessor:
    """Simple OCR processor for PDFs and images."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.reader = easyocr.Reader(self.config.languages, gpu=self._check_gpu())

    @staticmethod
    def _check_gpu() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def process_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> List[tuple]:
        """Process PDF file and extract text from pages.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to process (0-indexed), None for all

        Returns:
            List of tuples (page_number, text)
        """
        images = pdf_to_images(pdf_path, self.config.dpi, pages)
        results = []

        for image, page_num in images:
            processed = preprocess_image(image)
            ocr_results = self.reader.readtext(processed)

            # Extract text from results
            text = " ".join([result[1] for result in ocr_results if result[1].strip()])
            results.append((page_num, text))

        return results

    def process_image(self, image_path: str) -> str:
        """Process single image file.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """

        img = Image.open(image_path)
        image = np.array(img)
        processed = preprocess_image(image)
        ocr_results = self.reader.readtext(processed)

        return " ".join([result[1] for result in ocr_results if result[1].strip()])

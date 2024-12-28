import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from pdf2image import convert_from_path

from src.core.image_handler import ImageHandler

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Container for PDF page data and metadata.

    Attributes:
        image: Numpy array of the page image
        page_number: Page number in the PDF
        dpi: DPI used for conversion
        size: Tuple of (width, height) in pixels
    """

    image: npt.NDArray
    page_number: int
    dpi: int
    size: tuple[int, int]


class PDFHandler:
    """Handles PDF document processing and conversion to images.

    This class manages the conversion of PDF documents to images and coordinates
    with ImageHandler for preprocessing. It supports batch processing and
    maintains conversion metadata.

    Attributes:
        image_handler: ImageHandler instance for image processing
        _supported_pdf_versions: Set of supported PDF versions
    """

    _supported_pdf_versions = {"1.4", "1.5", "1.6", "1.7"}

    def __init__(
        self,
        image_handler: ImageHandler,
        max_workers: int = 4,
        memory_limit: int = 1024,  # MB
    ):
        """Initialize PDFHandler with configuration.

        Args:
            image_handler: Instance of ImageHandler for image processing
            max_workers: Maximum number of parallel workers
            memory_limit: Maximum memory usage in MB
        """
        self.image_handler = image_handler
        self.max_workers = max_workers
        self.memory_limit = memory_limit

    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        pages: Optional[List[int]] = None,
        batch_size: int = 10,
    ) -> List[PDFPage]:
        """Convert PDF file to list of images with metadata.

        Handles large PDFs by processing in batches to manage memory usage.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for conversion (higher means better quality but larger files)
            pages: Specific pages to convert (None means all pages)
            batch_size: Number of pages to process at once

        Returns:
            List[PDFPage]: List of converted pages with metadata

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is corrupted or unsupported
            MemoryError: If conversion exceeds memory limit
        """
        pdf_path = Path(pdf_path)
        self._validate_pdf(pdf_path)

        try:
            logger.info(f"Starting conversion of {pdf_path} at {dpi} DPI")

            # Convert pages in batches
            all_pages = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                page_batches = self._get_page_batches(pdf_path, batch_size, pages)

                for batch in page_batches:
                    future_pages = [
                        executor.submit(
                            self._convert_single_page, pdf_path, page_num, dpi
                        )
                        for page_num in batch
                    ]

                    for future in future_pages:
                        page = future.result()
                        all_pages.append(page)

            logger.info(f"Successfully converted {len(all_pages)} pages")
            return sorted(all_pages, key=lambda x: x.page_number)

        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
            raise

    def _validate_pdf(self, pdf_path: Path) -> None:
        """Validate PDF file before processing.

        Args:
            pdf_path: Path to PDF file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If PDF is invalid or unsupported
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # [TODO] Add PDF validation logic here
        # [TODO] Check PDF version, corruption, etc.

    def _convert_single_page(self, pdf_path: Path, page_num: int, dpi: int) -> PDFPage:
        """Convert a single PDF page to image.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to convert
            dpi: DPI for conversion

        Returns:
            PDFPage: Converted page with metadata
        """
        image = convert_from_path(
            pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
        )[0]

        np_image = np.array(image)
        return PDFPage(image=np_image, page_number=page_num, dpi=dpi, size=image.size)

    @staticmethod
    def _get_page_batches(
        pdf_path: Path, batch_size: int, pages: Optional[List[int]] = None
    ) -> List[List[int]]:
        """Split pages into batches for processing.

        Args:
            pdf_path: Path to PDF file
            batch_size: Size of each batch
            pages: Specific pages to process

        Returns:
            List of page number batches
        """
        if pages is None:
            # Get total page count
            total_pages = len(convert_from_path(pdf_path, dpi=72))
            pages = list(range(1, total_pages + 1))

        return [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]

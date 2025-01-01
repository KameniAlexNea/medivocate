import logging
from pathlib import Path
from typing import List, Optional, Union

from pdf2image import convert_from_path
from PIL import Image

from ..enums.ocr_enum import ImageFormat, Language


class PDFConverter:
    def __init__(
        self,
        languages: List[Language] = [Language.ENG],
        dpi: int = 300,
        fmt: ImageFormat = ImageFormat.PNG,
        grayscale: bool = False,
        use_threads: bool = False,
        thread_count: int = 4,
    ) -> None:
        """
        Initialize PDF converter with specified settings.

        Args:
            languages: List of languages to support
            dpi: Image resolution
            fmt: Output format
            grayscale: Convert to grayscale
            use_threads: Enable multi-threading
            thread_count: Number of threads for processing
        """
        self.languages = languages
        self.dpi = dpi
        self.fmt = fmt
        self.grayscale = grayscale
        self.use_threads = use_threads
        self.thread_count = thread_count
        self.logger = logging.getLogger(__name__)

    def convert_pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        first_page: int = 0,
        last_page: int = 0,
    ):
        """
        Convert PDF pages to images with enhanced options.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images
            first_page: Start page number
            last_page: End page number

        Returns:
            List of PIL Image objects
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            conversion_kwargs = {
                "dpi": self.dpi,
                "grayscale": self.grayscale,
                "thread_count": self.thread_count if self.use_threads else 1,
                "first_page": first_page,
                "last_page": last_page,
            }

            self.logger.info(f"Converting PDF: {pdf_path}")
            images = convert_from_path(pdf_path, **conversion_kwargs)

            if self.grayscale:
                images = [img.convert("L") for img in images]

            if output_dir:
                self._save_images(images, output_dir)

            return images

        except Exception as e:
            self.logger.error(f"Error converting PDF: {str(e)}")
            raise

    def _save_images(self, images: List[Image.Image], output_dir: Union[str, Path]):
        """Save converted images to specified directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(images, 1):
            image_path = output_dir / f"page_{i}.{self.fmt.value.lower()}"
            image.save(image_path, self.fmt.value)
            self.logger.debug(f"Saved image: {image_path}")

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return [lang.value for lang in self.languages]

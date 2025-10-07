from dataclasses import dataclass
from typing import List


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    languages: List[str] = None
    dpi: int = 300
    engine: str = "easyocr"  # "easyocr", "docling", or "pdf"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "fr"]

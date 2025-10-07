from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    languages: List[str] = None
    dpi: int = 300
    engine: str = "easyocr"  # "easyocr" or "docling"
    docling_format: str = "markdown"  # "markdown", "json", "text"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "fr"]

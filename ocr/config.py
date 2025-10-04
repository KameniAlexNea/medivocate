from dataclasses import dataclass
from typing import List


@dataclass
class OCRConfig:
    """Minimal configuration for OCR processing."""
    languages: List[str] = None
    dpi: int = 300

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']
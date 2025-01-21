from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ocr_enum import OutputFormat


@dataclass()
class PreprocessingConfig:
    """Configuration for image preprocessing steps."""

    denoise: bool = True
    deskew: bool = True
    contrast_enhancement: bool = True
    threshold: Optional[float] = None
    resize_factor: Optional[float] = None


@dataclass
class OCRConfig:
    """Main configuration class for OCR processing.

    Attributes:
        dpi: DPI for PDF to image conversion
        languages: List of language codes for OCR
        batch_size: Number of images to process in parallel
        preprocessing: Preprocessing configuration
        output_format: Desired output format
    """

    dpi: int # = 300
    languages: list[str] # = field(default_factory=lambda x: ["eng"])
    batch_size: int # = 10
    preprocessing: PreprocessingConfig # = PreprocessingConfig()
    output_format: OutputFormat = OutputFormat.TEXT

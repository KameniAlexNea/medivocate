from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ..enums.ocr_enum import OutputFormat


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
        temp_dir: Directory for temporary files
        output_dir: Directory for processed results
        dpi: DPI for PDF to image conversion
        languages: List of language codes for OCR
        batch_size: Number of images to process in parallel
        preprocessing: Preprocessing configuration
        output_format: Desired output format
    """

    temp_dir: Path = Path("./temp")
    output_dir: Path = Path("./output")
    dpi: int = 300
    languages: list[str] = None
    batch_size: int = 10
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    output_format: OutputFormat = OutputFormat.JSON

    def __post_init__(self):
        """Ensure directories exist and languages are set."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.languages is None:
            self.languages = ["eng"]

    def test(self):
        print("test")

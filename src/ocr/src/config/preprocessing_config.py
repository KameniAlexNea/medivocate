from dataclasses import dataclass
from typing import Optional


@dataclass()
class PreprocessingConfig:
    """Configuration for image preprocessing steps."""

    denoise: bool = True
    deskew: bool = True
    contrast_enhancement: bool = True
    threshold: Optional[float] = None
    resize_factor: Optional[float] = None

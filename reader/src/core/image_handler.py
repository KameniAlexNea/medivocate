import logging
from pathlib import Path
from typing import Union

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.config.ocr_config import PreprocessingConfig
from src.utils.preprocessing import ImagePreprocessor, ProcessedImage

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles all image-related operations including loading and preprocessing.

    This class serves as a facade for image processing operations, coordinating
    between different preprocessing steps and ensuring proper image handling.

    Attributes:
        preprocessor: Instance of ImagePreprocessor
        _supported_formats: Set of supported image formats
    """

    _supported_formats = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    def __init__(self, preprocessor: ImagePreprocessor):
        """Initialize ImageHandler with preprocessor.

        Args:
            preprocessor: Instance of ImagePreprocessor for image enhancement
        """
        self.preprocessor = preprocessor

    def load_image(self, path: Union[str, Path]) -> npt.NDArray:
        """Load image from file path with validation.

        Args:
            path: Path to image file

        Returns:
            numpy.ndarray: Loaded image

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.suffix.lower() not in self._supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {self._supported_formats}"
            )

        try:
            img = Image.open(path)
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image {path}: {str(e)}")
            raise

    def preprocess_image(
        self, image: npt.NDArray, config: PreprocessingConfig
    ) -> ProcessedImage:
        """Apply preprocessing steps based on configuration.

        Applies a series of image enhancement techniques to improve OCR accuracy.

        Args:
            image: Input image as numpy array
            config: Preprocessing configuration

        Returns:
            ProcessedImage: Processed image with metadata

        Raises:
            ValueError: If preprocessing fails
        """
        try:
            result = ProcessedImage(image=image)

            if config.denoise:
                denoised = self.preprocessor.denoise(result.image)
                result.image = denoised.image
                result.preprocessing_history.extend(denoised.preprocessing_history)

            if config.deskew:
                deskewed = self.preprocessor.deskew(result.image)
                result.image = deskewed.image
                result.angle = deskewed.angle
                result.preprocessing_history.extend(deskewed.preprocessing_history)

            return result

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

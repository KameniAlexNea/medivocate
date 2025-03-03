import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    """Container for processed image and its metadata."""

    image: npt.NDArray
    angle: Optional[float] = None
    enhancement_applied: bool = False
    preprocessing_history: Optional[list[str]] = None

    def __post_init__(self):
        """Initialize preprocessing history if not provided."""
        if self.preprocessing_history is None:
            self.preprocessing_history = []


class ImagePreprocessor:
    """Handles all image preprocessing operations.

    This class provides methods to enhance image quality for better OCR results.
    Each method is designed to be independent and chainable.
    """

    @staticmethod
    def denoise(image: npt.NDArray) -> ProcessedImage:
        """Remove noise from image using Non-Local Means Denoising.

        Args:
            image: Input image as numpy array

        Returns:
            ProcessedImage: Processed image with metadata

        Raises:
            ValueError: If input image is invalid
        """
        try:
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

            return ProcessedImage(image=denoised, preprocessing_history=["denoise"])
        except Exception as e:
            logger.error(f"Error during denoising: {str(e)}")
            raise ValueError(f"Failed to denoise image: {str(e)}")

    @staticmethod
    def deskew(image: npt.NDArray) -> ProcessedImage:
        """Correct image skew by detecting and rotating to align text.

        Uses contour detection to find the dominant text angle and corrects it.

        Args:
            image: Input image as numpy array

        Returns:
            ProcessedImage: Deskewed image with rotation angle

        Raises:
            ValueError: If angle detection fails
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect edges
            edges = cv2.Canny(gray, 50, 200, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is None:
                logger.warning("No lines detected for deskewing")
                return ProcessedImage(image=image, angle=0)

            # Calculate dominant angle
            angles = []
            for _, theta in lines[0]:
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)

            if not angles:
                return ProcessedImage(image=image, angle=0)

            median_angle = np.median(angles)

            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            return ProcessedImage(
                image=rotated, angle=median_angle, preprocessing_history=["deskew"]
            )
        except Exception as e:
            logger.error(f"Error during deskewing: {str(e)}")
            raise ValueError(f"Failed to deskew image: {str(e)}")

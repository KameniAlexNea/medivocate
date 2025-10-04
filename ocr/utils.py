import io
import logging
from typing import List, Optional

import cv2
import numpy as np
import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str, dpi: int = 300, pages: Optional[List[int]] = None
) -> List[tuple]:
    """Convert PDF pages to images.

    Args:
        pdf_path: Path to PDF file
        dpi: DPI for conversion
        pages: List of page numbers to convert (0-indexed), None for all

    Returns:
        List of tuples (image_array, page_number)
    """
    doc = pymupdf.open(pdf_path)
    if pages is None:
        pages = list(range(len(doc)))

    images = []
    for page_num in pages:
        page = doc[page_num]
        pix = pymupdf.utils.get_pixmap(page, dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes()))
        images.append((np.array(img), page_num))
    doc.close()
    return images


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Apply basic preprocessing to improve OCR accuracy.

    Args:
        image: Input image as numpy array

    Returns:
        Processed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Deskew
    edges = cv2.Canny(denoised, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        angles = []
        for _, theta in lines[0]:
            angle = theta * 180 / np.pi
            if angle < 45:
                angles.append(angle)
            elif angle > 135:
                angles.append(angle - 180)

        if angles:
            median_angle = np.median(angles)
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            denoised = cv2.warpAffine(
                denoised,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    return denoised

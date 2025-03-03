import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pymupdf
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    image: npt.NDArray
    page_number: int
    dpi: int
    size: tuple[int, int]


def get_page_data(page: pymupdf.Document, page_id, dpi, resize=False):
    img = Image.open(io.BytesIO(pymupdf.utils.get_pixmap(page, dpi=dpi).tobytes()))
    if resize:
        img = img.resize(size=(800, 1333))  # fixed shape for batch ocr engine ?
    return PDFPage(np.array(img), page_id, dpi, img.size)


class PDFHandler:

    def pdf_to_images_batch(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 300,
        pages: Optional[List[int]] = None,
        batch_size: int = 8,
    ):
        pdf_path = Path(pdf_path)

        doc = pymupdf.open(pdf_path)
        if pages is None:
            pages = list(range(len(doc)))
        page_batched = [
            pages[i : i + batch_size] for i in range(0, len(pages), batch_size)
        ]
        for batch in page_batched:
            yield [get_page_data(doc[i], i, dpi, not (batch_size == 1)) for i in batch]

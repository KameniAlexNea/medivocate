import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.enums.ocr_enum import ImageFormat, Language
from src.utils.pdf_converter import PDFConverter


@pytest.fixture
def sample_pdf():
    return Path(__file__).parent / "test_files/sample.pdf"


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def converter():
    return PDFConverter(languages=[Language.ENG], dpi=200, fmt=ImageFormat.PNG)


def test_pdf_conversion(converter: PDFConverter, sample_pdf, output_dir):
    images = converter.convert_pdf_to_images(sample_pdf, output_dir)
    assert len(images) > 0
    assert all(isinstance(img, Image.Image) for img in images)
    assert len(list(output_dir.glob("*.png"))) == len(images)


def test_grayscale_conversion(sample_pdf, output_dir):
    converter = PDFConverter(grayscale=True)
    images = converter.convert_pdf_to_images(sample_pdf, output_dir)
    assert all(img.mode == "L" for img in images)


def test_language_support():
    langs = [Language.ENG, Language.FRA]
    converter = PDFConverter(languages=langs)
    assert len(converter.get_supported_languages()) == len(langs)


def test_invalid_pdf_path(converter: PDFConverter):
    with pytest.raises(FileNotFoundError):
        converter.convert_pdf_to_images("nonexistent.pdf")


def test_page_range_conversion(converter: PDFConverter, sample_pdf):
    images = converter.convert_pdf_to_images(sample_pdf, first_page=1, last_page=2)
    assert len(images) <= 2


def test_different_formats(sample_pdf, output_dir):
    for fmt in ImageFormat:
        converter = PDFConverter(fmt=fmt)
        converter.convert_pdf_to_images(sample_pdf, output_dir)
        assert len(list(output_dir.glob(f"*.{fmt.value.lower()}"))) > 0


def test_thread_count(sample_pdf):
    thread_counts = [1, 2, 4]
    for count in thread_counts:
        converter = PDFConverter(use_threads=True, thread_count=count)
        images = converter.convert_pdf_to_images(sample_pdf)
        assert len(images) > 0

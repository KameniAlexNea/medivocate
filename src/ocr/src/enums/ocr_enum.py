from enum import Enum


class Language(Enum):
    ENG = "en"
    FRA = "fra"
    DEU = "deu"
    SPA = "spa"
    ITA = "ita"
    JPN = "jpn"
    KOR = "kor"
    RUS = "rus"
    ARA = "ara"


class ImageFormat(Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"


class OutputFormat(Enum):
    """Supported output formats for OCR results."""

    JSON = "json"
    XML = "xml"
    TEXT = "text"

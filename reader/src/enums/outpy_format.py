from enum import Enum


class OutputFormat(Enum):
    """Supported output formats for OCR results."""

    JSON = "json"
    XML = "xml"
    TEXT = "text"

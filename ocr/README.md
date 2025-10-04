# OCR Package

A simple OCR package for extracting text from PDF files using EasyOCR.

## Features

- Convert PDF pages to images
- Apply basic image preprocessing (denoising, deskewing)
- Extract text using EasyOCR
- Support for multiple languages
- Output text files per page

## Usage

### Command Line

```bash
python -m ocr.main --pdf_path /path/to/document.pdf --output_folder /path/to/output
```

### Python API

```python
from ocr import OCRConfig, OCRProcessor

config = OCRConfig(languages=['en', 'fr'], dpi=300)
processor = OCRProcessor(config)

results = processor.process_pdf('document.pdf')
for page_num, text in results:
    print(f"Page {page_num}: {text}")
```

## Configuration

- `languages`: List of language codes (default: ['en'])
- `dpi`: DPI for PDF to image conversion (default: 300)

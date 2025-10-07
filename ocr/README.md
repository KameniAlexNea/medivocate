# OCR Package

A flexible OCR package for extracting text from PDF files and images using multiple engines.

## Features

- Support for multiple OCR engines: EasyOCR and Docling
- Convert PDF pages to images (EasyOCR)
- Apply basic image preprocessing (denoising, deskewing) for EasyOCR
- Extract text using EasyOCR or Docling
- Support for multiple languages (EasyOCR)
- Multiple output formats (Docling)
- Output text files per page

## Usage

### Command Line

```bash
# Using EasyOCR (default)
python -m ocr.main --pdf_path /path/to/document.pdf --output_folder /path/to/output

# Using Docling
python -m ocr.main --pdf_path /path/to/document.pdf --output_folder /path/to/output --engine docling
```

### Python API

```python
from ocr import OCRConfig, EasyOCRProcessor, DoclingOCRProcessor

# Using EasyOCR
config = OCRConfig(languages=['en', 'fr'], dpi=300, engine='easyocr')
processor = EasyOCRProcessor(config)

results = processor.process_pdf('document.pdf')
for page_num, text in results:
    print(f"Page {page_num}: {text}")

# Using Docling
config = OCRConfig(engine='docling')
processor = DoclingOCRProcessor(config)

results = processor.process_pdf('document.pdf')
for page_num, text in results:
    print(f"Page {page_num}: {text}")
```

## Configuration

- `languages`: List of language codes for EasyOCR (default: ['en', 'fr'])
- `dpi`: DPI for PDF to image conversion for EasyOCR (default: 300)
- `engine`: OCR engine to use - 'easyocr' or 'docling' (default: 'easyocr')

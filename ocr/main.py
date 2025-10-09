import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from .processors import (
    DoclingOCRProcessor,
    EasyOCRProcessor,
    OCRConfig,
    PDFProcessor,
)


def process_document(
    file_path: str, output_folder: str, processor
):
    """
    Process a document with OCR and save the result to the output folder.

    Args:
        file_path: Path to the input PDF file
        output_folder: Destination folder for the output files
        config: OCR configuration
    """
    if not file_path.lower().endswith(".pdf"):
        logger.warning(f"Skipping non-PDF file: {file_path}")
        return

    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Process PDF
        results = processor.process_pdf(file_path)

        # Save results
        base_name = Path(file_path).stem
        for page_num, text in results:
            output_file_path = os.path.join(
                output_folder, f"{base_name}-page_{page_num:04d}.txt"
            )
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(text)
            logger.info(f"Output saved to: {output_file_path}")

    except Exception as e:
        logger.error(f"Error processing file '{file_path}': {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description="OCR PDF Document")
    parser.add_argument(
        "--pdf_path",
        required=True,
        type=str,
        help="Path to the PDF file or folder containing PDF files",
    )
    parser.add_argument(
        "--dpi",
        default=300,
        type=int,
        help="DPI (dots per inch) for OCR processing (default: 300)",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        type=str,
        help="Output folder for OCR results",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "fr"],
        help="Languages for OCR (default: en)",
    )
    parser.add_argument(
        "--engine",
        choices=["easyocr", "docling", "pdf"],
        default="easyocr",
        help="OCR engine to use (default: easyocr)",
    )

    args = parser.parse_args()

    config = OCRConfig(languages=args.languages, dpi=args.dpi, engine=args.engine)

    # Create processor once
    if config.engine == "docling":
        processor = DoclingOCRProcessor(config)
    elif config.engine == "easyocr":
        processor = EasyOCRProcessor(config)
    else:  # config.engine == "pdf"
        processor = PDFProcessor(config)

    if os.path.isfile(args.pdf_path):
        process_document(args.pdf_path, args.output_folder, processor)
    else:
        pdf_files = glob(os.path.join(args.pdf_path, "*.pdf"))
        if not pdf_files:
            logger.error("No PDF files found in the specified folder")
            exit(1)

        for pdf_file in tqdm(pdf_files):
            output_subfolder = os.path.join(args.output_folder, Path(pdf_file).stem)
            if (
                os.path.exists(output_subfolder)
                and len(os.listdir(output_subfolder)) > 0
            ):
                continue
            process_document(pdf_file, output_subfolder, processor)

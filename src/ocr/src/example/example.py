# examples/usage_example.py
import logging

from src.config.ocr_config import OCRConfig
from src.config.preprocessing_config import PreprocessingConfig
from src.core.image_handler import ImageHandler
from src.core.ocr_engine import OCREngine
from src.core.pdf_handler import PDFHandler
from src.enums.language import Language
from src.enums.outpy_format import OutputFormat
from src.utils.preprocessing import ImagePreprocessor


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def process_document(file_path: str, output_format: OutputFormat = OutputFormat.TEXT):
    """Process a document with OCR and display results.

    Args:
        file_path: Path to PDF or image file
        output_format: Desired output format
    """
    # Initialize configuration
    config = OCRConfig(
        dpi=300,
        languages=["en"],  # Add more languages as needed
        batch_size=4,
        preprocessing=PreprocessingConfig(
            denoise=True, deskew=True, contrast_enhancement=True
        ),
        output_format=output_format,
    )

    # Initialize components
    preprocessor = ImagePreprocessor()
    image_handler = ImageHandler(preprocessor)
    pdf_handler = PDFHandler(
        image_handler=image_handler, max_workers=4, memory_limit=1024
    )

    # Initialize OCR engine
    ocr_engine = OCREngine(
        pdf_handler=pdf_handler, image_handler=image_handler, config=config
    )

    try:
        # Process file
        result = ocr_engine.process_file(file_path, output_format)

        # Handle different output formats
        if output_format == OutputFormat.TEXT:
            print("\nExtracted Text:")
            print(result)
        elif output_format == OutputFormat.JSON:
            print("\nJSON Output:")
            print(result)
        else:
            print("\nXML Output:")
            print(result)

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise


def main():
    """Main function demonstrating OCR usage."""
    setup_logging()

    # Example usage with different file types and output formats
    examples = [
        {
            "file": "src/samples/fable.pdf",
            "format": OutputFormat.TEXT,
        },
        # {"file": "../samples/letter.jpg", "format": OutputFormat.JSON},
    ]

    for example in examples:
        print(f"\nProcessing: {example['file']}")
        print(f"Output format: {example['format'].value}")
        try:
            process_document(example["file"], example["format"])
        except Exception as e:
            print(f"Failed to process {example['file']}: {str(e)}")


if __name__ == "__main__":
    main()

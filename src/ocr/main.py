import logging
import os

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


def process_document(file_path: str, output_format: OutputFormat, output_folder: str):
    """
    Process a document with OCR and save the result to the output folder.

    Args:
        file_path (str): Path to the input PDF or image file.
        output_format (OutputFormat): Desired output format (TEXT, JSON, XML).
        output_folder (str): Destination folder for the output file.
    """
    # Setup logging
    setup_logging()

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

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Prepare output file name and path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extension = output_format.value.lower()
        output_file_path = os.path.join(
            output_folder, f"{base_name}.{extension}")

        # Write results to file
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(result)

        logging.info(f"Output saved to: {output_file_path}")

    except Exception as e:
        logging.error(f"Error processing file '{file_path}': {str(e)}")
        raise

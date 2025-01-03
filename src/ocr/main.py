import logging
import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

from .config.ocr_config import OCRConfig, PreprocessingConfig
from .core.ocr_base_engine import OCREngine
from .enums.ocr_enum import OutputFormat


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
    ocr_engine = OCREngine(config)

    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        # Process file
        for pages, contents in ocr_engine.convert_pdf_to_data(
            file_path, list(range(16))
        ):
            for page_num, result in zip(pages, contents):
                # Prepare output file name and path
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                extension = output_format.value.lower()
                output_file_path = os.path.join(
                    output_folder, f"{base_name}-page_{page_num:04d}.{extension}"
                )

                # Write results to file
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(result)

                logging.info(f"Output saved to: {output_file_path}")

    except Exception as e:
        logging.error(f"Error processing file '{file_path}': {str(e)}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser(description="OCRize PDF Document")
    parser.add_argument(
        "--inputs",
        required=False,
        type=str,
        help="Path to the file or folder containing list of files",
    )
    parser.add_argument(
        "--outputs",
        default="text",
        required=False, # output format is computed
        type=str,
        help="output format of OCR Document.",
    )

    args = parser.parse_args()

    if os.path.isfile(args.inputs):
        output_folder = args.inputs.replace(".pdf", "")
        process_document(
            args.inputs, OutputFormat[args.outputs.upper()], output_folder
        )
    else:
        files = glob(os.path.join(args.inputs, "*.pdf"))
        assert len(files), "At least one file in the folder passed"
        for file in tqdm(files):
            os.makedirs(file.replace(".pdf", ""), exist_ok=True)
            process_document(
                file, OutputFormat[args.outputs.upper()], file.replace(".pdf", "")
            )

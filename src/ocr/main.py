import logging
import os
from argparse import ArgumentParser
from glob import glob

from tqdm import tqdm

from .config.ocr_config import OCRConfig, PreprocessingConfig
from .config.ocr_enum import OutputFormat
from .core.ocr_base_engine import OCREngine


def process_document(file_path: str, output_format: OutputFormat, output_folder: str):
    """
    Process a document with OCR and save the result to the output folder.

    Args:
        file_path (str): Path to the input PDF or image file.
        output_format (OutputFormat): Desired output format (TEXT, JSON, XML).
        output_folder (str): Destination folder for the output file.
    """
    test_pdf_readable = len(glob(os.path.join(output_folder, "*.txt"))) > 2
    if test_pdf_readable:
        logging.info(f"Skipping {file_path} as it is already processed")
        return
    logging.info(f"Processing {file_path}")

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
        "--pdf_path",
        required=False,
        type=str,
        help="Path to the file or folder containing list of files",
    )
    parser.add_argument(
        "--output_type",
        default="text",
        required=False,  # output format is computed
        type=str,
        help="output format of OCR Document.",
    )

    args = parser.parse_args()

    if os.path.isfile(args.pdf_path):
        output_folder = args.pdf_path.replace(".pdf", "")
        os.makedirs(output_folder, exist_ok=True)
        process_document(
            args.pdf_path, OutputFormat[args.output_type.upper()], output_folder
        )
    else:
        files = glob(os.path.join(args.pdf_path, "*.pdf"))
        assert len(files), "At least one file in the folder passed"
        for file in tqdm(files):
            output_folder = file.replace(".pdf", "")
            os.makedirs(output_folder, exist_ok=True)
            process_document(
                file, OutputFormat[args.output_type.upper()], output_folder
            )

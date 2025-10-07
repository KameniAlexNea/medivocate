import os
import shutil
import subprocess
import zipfile

from loguru import logger


def download_and_prepare_data(gdrive_url, zip_filename, extract_to, target_folder):
    """
    Download, extract, and organize data from a Google Drive link.

    :param gdrive_url: Google Drive URL to download the zip file.
    :param zip_filename: Name for the downloaded zip file.
    :param extract_to: Directory to extract the zip file.
    :param target_folder: Final directory to move extracted content.
    """
    try:
        if os.path.exists(os.path.join(target_folder, "chroma.sqlite3")):
            logger.info(f"Data already exists in {target_folder}")
            return
        # Step 1: Download the file using gdown
        logger.info("Downloading file...")
        subprocess.run(["gdown", gdrive_url, "-O", zip_filename], check=True)

        # Step 2: Unzip the downloaded file
        logger.info("Unzipping file...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        # Step 3: Remove old data folder if it exists
        if os.path.exists(target_folder):
            logger.info(f"Removing existing folder: {target_folder}")
            shutil.rmtree(target_folder)

        # Step 4: Move the extracted folder to the target location
        logger.info(f"Moving extracted data to {target_folder}")
        extracted_folder = os.path.join(extract_to, os.path.basename(target_folder))
        shutil.move(extracted_folder, target_folder)

        # Step 5: Remove the downloaded zip file
        logger.info(f"Cleaning up, removing zip file: {zip_filename}")
        os.remove(zip_filename)

        logger.info("Data preparation completed successfully!")
    except Exception as e:
        logger.info(f"An error occurred: {e}")

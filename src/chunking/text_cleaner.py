import os

from tqdm import tqdm

from .processor import Processor


def clean_text(text):
    # Implement your text cleaning logic here
    cleaned_text = Processor.merge_sentences(
        text
    )  # Example: stripping leading/trailing whitespace
    return cleaned_text


def process_text_files(folder_path: str):
    for subdir, _, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                cleaned_text = clean_text(text)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Clean OCRized Documents")
    parser.add_argument(
        "--pdf_text_path",
        required=False,
        type=str,
        help="Path to the file or folder containing list of files",
    )
    args = parser.parse_args()
    process_text_files(args.pdf_text_path)

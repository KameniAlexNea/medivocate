import argparse
import os
import random
import uuid
from glob import glob

from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_chat
from .prompts import OPEN_QUESTION_PROMPT


def generate_questions(input_folder: str, n_files: int, output_folder: str):
    """
    Generate questions using an LLM based on text files in a folder and save the results in a specified folder.

    Args:
        input_folder (str): Path to the folder containing input text files.
        n_files (int): Number of files to process.
        output_folder (str): Path to the folder where output files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob(os.path.join(input_folder, "*.txt")) + glob(
        os.path.join(input_folder, "*/*.txt")
    )
    print(f"Found {len(files)} files in {input_folder}.")

    if len(files):
        files = random.sample(files, min(n_files, len(files)))

        llm = get_llm_model_chat("OLLAMA", temperature=0.1, max_tokens=1000)

        for file in tqdm(files):
            lines = open(file).read()
            if len(lines.strip().split()) < 100:
                print(f"Ignoring {file} (too few words)")
                continue

            # Generate the text using the LLM
            text = llm.invoke([("system", OPEN_QUESTION_PROMPT.format(context=lines))])

            # Generate a unique filename for the output
            output_filename = f"{uuid.uuid4().hex}.txt"
            output_path = os.path.join(output_folder, output_filename)

            # Save the generated content to the output file
            with open(output_path, "w") as out_file:
                out_file.write(text.content)

            print(f"Saved generated questions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing input text files.",
    )
    parser.add_argument("--n_files", type=int, help="Number of files to process.")
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where output files will be saved.",
    )

    args = parser.parse_args()

    generate_questions(args.input_folder, args.n_files, args.output_folder)

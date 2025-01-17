"""
python -m src.llm_evaluation.llm_validator --prediction_folder data/llm_eval_predictions --query_folder data/clear_evaluation --output_folder data/evaluation_results
"""

import argparse
import json
import os
from glob import glob

from tqdm import tqdm

from ..utilities.llm_models import get_llm_model_chat
from .prompts import VALIDATOR_PROMPT_FR, VALIDATOR_PROMPT_FR_CONTENT


def generate_questions(
    prediction_folder: str,
    query_folder: str,
    output_folder: str,
):
    """
    Generate questions using an LLM based on text files in a folder and save the results in a specified folder.

    Args:
        input_folder (str): Path to the folder containing input text files.
        n_files (int): Number of files to process.
        output_folder (str): Path to the folder where output files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files: list[str] = glob(os.path.join(prediction_folder, "*.txt"))

    llm = get_llm_model_chat(temperature=0.1, max_tokens=1000)

    for file in tqdm(files):
        preds = open(file).read().strip()
        if not len(preds):
            print("Ignore file due to empty prediction", file)
            continue
        name = os.path.basename(file).replace(".txt", ".json")
        raw: dict[str] = json.load(open(os.path.join(query_folder, name)))

        # Generate the text using the LLM
        text = llm.invoke(
            [
                ("system", VALIDATOR_PROMPT_FR),
                (
                    "user",
                    VALIDATOR_PROMPT_FR_CONTENT.format(
                        question=raw["question"], answer=raw["answer"], suggested=preds
                    ),
                ),
            ]
        )
        result = {"evaluation": text.content.strip(), **raw, "suggested": preds}

        output_filename = name
        output_path = os.path.join(output_folder, output_filename)

        # Save the generated content to the output file
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(result, out_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--prediction_folder",
        type=str,
        help="Path to the folder containing input text files of questions",
    )
    parser.add_argument(
        "--query_folder",
        type=str,
        help="Path to the folder where output files will be saved.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where output files will be saved.",
    )

    args = parser.parse_args()

    generate_questions(args.prediction_folder, args.query_folder, args.output_folder)

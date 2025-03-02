SYSTEM_PROMPT = """You are given a long text. Your task is to split this text into several chunks, where each chunk represents one main idea of the text. Ensure that all parts of the text are covered with no overlapping ideas.

For each chunk, do the following:
- Provide a short classification (in a few words) that describes what the chunk is about.
- Ensure that the classification captures the essence of the chunk's main idea.

Return the result as a list of dictionaries, where each dictionary has two keys:
- `"chunk"`: the text corresponding to the main idea.
- `"classification"`: a brief summary of the topic covered by the chunk.

For context, the text comes from a source related to the Medivocate app, which is described as follows:

> "Medivocate is an application that offers clear and structured information about African history and traditional medicine. The knowledge is exclusively based on historical documentaries about the African continent."

Make sure your output follows this structure and clearly segments the text into its main ideas with corresponding classifications.

Example Output Structure:
```
[
    {"chunk": "Les Ta’rīkh nous donnent la liste des dignitaires du pouvoir central dont nous retenons les princi", "classification": "Dignitaires du pouvoir central"},
    {"chunk": "Le « fari mondzo» ou « monjo » était le ministre de l’agriculture. Il est très possible qu’il se soit occupé de la direction de", "classification": "Ministre de l'agriculture"},
]
```
"""

import argparse
import json
import os
import time
from glob import glob

import tqdm
from ..utilities.llm_models import get_llm_model_chat


class TextCleaner:
    def __init__(self):
        self.llm = get_llm_model_chat(temperature=.3, max_tokens=None)

    def prepare_text(self, text: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

    def clean_text(self, text):
        return self.llm.invoke(self.prepare_text(text)).content.strip()

    def clean_texts(self, texts):
        return [
            i.content.strip()
            for i in self.llm.batch([self.prepare_text(text) for text in texts])
        ]

    def run_chunks_processing(self, folder: str, batch_size: int = 4):
        files = glob(os.path.join(folder, "*.json"))
        save_folder = os.path.join(folder, "cleaned")
        files = [
            i
            for i in files
            if not os.path.isfile(
                os.path.join(save_folder, os.path.basename(i).replace(".json", ".txt"))
            )
        ]
        os.makedirs(save_folder, exist_ok=True)
        for i in tqdm.tqdm(range(0, len(files), batch_size)):
            batch = files[i : i + batch_size]
            raws = [json.load(open(f))["kwargs"]["page_content"] for f in batch]
            cleaned = []
            try:
                cleaned = self.clean_texts(raws)
                time.sleep(1)
            except Exception as ex:
                time.sleep(1)
                cleaned = self.clean_texts(raws)
                time.sleep(1)
            for f, c in zip(batch, cleaned):
                with open(
                    os.path.join(
                        save_folder, os.path.basename(f).replace(".json", ".txt")
                    ),
                    "w",
                    encoding="utf-8",
                ) as opf:
                    opf.write(
                        c,
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process JSON files in folder and clean texts."
    )
    parser.add_argument(
        "--folder",
        default="data/chunks",
        type=str,
        help="Path to folder containing .json files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing"
    )
    args = parser.parse_args()

    cleaner = TextCleaner()
    cleaner.run_chunks_processing(args.folder, batch_size=args.batch_size)

SYSTEM_PROMPT = """You are an AI assistant tasked with analyzing and segmenting text related to the Medivocate app. Medivocate is an application that offers clear and structured information about African history and traditional medicine. The knowledge is exclusively based on historical documentaries about the African continent.

Your task is to split this text into several chunks, where each chunk represents one main idea of the text. Follow these steps:

1. Read through the entire text carefully.
2. Identify the main ideas or topics discussed in the text.
3. Split the text into chunks, with each chunk corresponding to one main idea.
4. Ensure that all parts of the text are covered with no overlapping ideas.
5. Make sure that each chunk is coherent and self-contained.

For each chunk you create, do the following:
- Provide a short classification (in a few words) that describes what the chunk is about.
- Ensure that the classification captures the essence of the chunk's main idea.
- The classification should be concise but informative, allowing readers to quickly understand the topic of the chunk.

Return your result as a list of dictionaries. Each dictionary should have two keys:
- "chunk": the text corresponding to the main idea.
- "classification": a brief summary of the topic covered by the chunk.

Your output should follow this structure:

```
[
    {"chunk": "Text of the first chunk...", "classification": "Brief classification of first chunk"},
    {"chunk": "Text of the second chunk...", "classification": "Brief classification of second chunk"},
    ...
]
```

Additional guidelines:
- Preserve the original language of the text in your chunks.
- Keep the chunks to a reasonable size, typically a paragraph or two.
- Ensure that your classifications are relevant to the context of African history and traditional medicine.
- If you encounter any terms or concepts specific to African culture or history, include them in your classifications when appropriate.

Remember, the goal is to create a clear and structured segmentation of the text that would be useful for users of the Medivocate app to navigate and understand the content.
"""

import argparse
import json
import os
import time
from glob import glob

import tqdm

from ...utilities.llm_models import get_llm_model_chat


class TextCleaner:
    def __init__(self):
        self.llm = get_llm_model_chat(temperature=0.3, max_tokens=None)

    def prepare_text(self, text: str):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": """
Here is the text to split:

<text>
{TEXT}
</text>""".format(
                    TEXT=text
                ),
            },
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
            except Exception:
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

FORMAT_OUTPUT = """
```
[
    {"chunk": "Text of the first chunk...", "classification": "Brief classification of first chunk"},
    {"chunk": "Text of the second chunk...", "classification": "Brief classification of second chunk"},
    ...
]
```
"""

PARENT_SYSTEM = """You are an AI assistant tasked with analyzing and segmenting a given text into coherent chunks, each representing a main idea or topic. Your goal is to create a clear and structured segmentation of the text that helps readers navigate and understand the content, regardless of the subject."""

USER_PROMPT = """Here is the text you need to analyze and segment:

<text>
{TEXT}
</text>

Follow these steps to complete the task:

1. Read the entire text carefully to understand its content and structure.

2. Identify the main ideas or key topics discussed in the text.

3. Split the text into chunks, where each chunk corresponds to a single main idea. Aim for chunks that are typically one to two paragraphs long, ensuring they are neither too brief nor overly lengthy.

4. For each chunk you create:
   a. Provide a short classification (in a few words) summarizing what the chunk is about.
   b. Ensure the classification reflects the chunk's main idea accurately.
   c. Keep the classification concise and informative, allowing readers to quickly grasp the chunk's topic.

5. Ensure that all parts of the text are included without overlapping ideas between chunks.

6. Make sure each chunk is self-contained and makes sense independently.

7. Return your result as a `list` of `dictionaries` (as in python language), each with two keys:
   - "chunk": The text segment corresponding to the main idea.
   - "classification": A brief summary of the topic covered by the chunk.

The output should follow this structure:

<output>
{FORMAT_OUTPUT}
</output>

Additional guidelines:
- Preserve the original language of the text within the chunks; do not alter the wording.
- Ensure classifications are appropriate to the text's subject matter.
- If the text includes terms or concepts unique to its subject, include them in the classifications when relevant.

Remember, your objective is to create a clear and structured segmentation of the text that helps readers navigate and understand the content, regardless of the subject. Provide your answer in the specified JSON format inside <answer> tags.
"""

import argparse
import json
import os
from glob import glob

import tqdm
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


def get_llm_model_chat(temperature=0.01, max_tokens: int = None):
    return ChatAnthropic(model="claude-3-7-sonnet-20250219", max_tokens=30000)
    if str(os.getenv("USE_OLLAMA_CHAT")) == "1":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL"),
            temperature=temperature,
            num_predict=max_tokens,
        )
    return ChatGroq(
        model=os.getenv("GROQ_MODEL_NAME"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TextCleaner:
    def __init__(self):
        self.llm = get_llm_model_chat(temperature=0.3, max_tokens=None)

    def prepare_text(self, text: str):
        return [
            {"role": "system", "content": PARENT_SYSTEM},
            {
                "role": "user",
                "content": USER_PROMPT.format(TEXT=text, FORMAT_OUTPUT=FORMAT_OUTPUT),
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
                # time.sleep(1)
            except Exception:
                # time.sleep(1)
                cleaned = self.clean_texts(raws)
                # time.sleep(1)
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

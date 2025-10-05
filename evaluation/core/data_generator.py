"""Data generation for evaluation."""
import json
import os
import random
import uuid
from typing import List

from tqdm import tqdm

from ..config import EvaluationConfig
from ..models.evaluation_data import QAPair
from ..utils.file_utils import find_files_by_pattern, load_json_file, load_text_file, save_json_file
from ..utils.llm_utils import get_llm_client
from ..prompts import OPEN_QUESTION_PROMPT


class DataGenerator:
    """Generate evaluation data from documents."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = get_llm_client(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    def generate_evaluation_data(self) -> List[QAPair]:
        """Generate Q&A pairs from documents."""
        files = self._find_input_files()
        selected_files = self._select_files(files)

        qa_pairs = []
        for file_path in tqdm(selected_files, desc="Generating Q&A pairs"):
            try:
                content = self._load_file_content(file_path)
                if self._is_content_valid(content):
                    pairs = self._generate_qa_from_content(content, file_path)
                    qa_pairs.extend(pairs)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return qa_pairs

    def _find_input_files(self) -> List[str]:
        """Find all input files."""
        pattern = f"*.{self.config.file_type}"
        return find_files_by_pattern(self.config.input_folder, pattern)

    def _select_files(self, files: List[str]) -> List[str]:
        """Select subset of files for processing."""
        if len(files) <= self.config.n_files:
            return files
        return random.sample(files, self.config.n_files)

    def _load_file_content(self, file_path: str) -> str:
        """Load content from file."""
        if file_path.endswith('.txt'):
            return load_text_file(file_path)
        elif file_path.endswith('.json'):
            data = load_json_file(file_path)
            return data.get("kwargs", {}).get("page_content", str(data))
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _is_content_valid(self, content: str) -> bool:
        """Check if content meets minimum requirements."""
        word_count = len(content.split())
        return word_count >= self.config.min_words_threshold

    def _generate_qa_from_content(self, content: str, file_path: str) -> List[QAPair]:
        """Generate Q&A pairs from content using LLM."""
        prompt = OPEN_QUESTION_PROMPT.format(content=content[:2000])  # Limit content length

        response = self.llm.invoke([
            ("system", "You are a helpful assistant that generates questions and answers from text."),
            ("user", prompt)
        ])

        # Parse the response (assuming it returns structured data)
        try:
            qa_data = self._parse_llm_response(response.content)
            qa_pairs = []

            for qa_dict in qa_data:
                qa_pair = QAPair.from_dict(qa_dict)
                qa_pair.metadata = {
                    "source_file": file_path,
                    "generated_id": str(uuid.uuid4()),
                    **qa_pair.metadata
                }
                qa_pairs.append(qa_pair)

            return qa_pairs
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def _parse_llm_response(self, response: str) -> List[dict]:
        """Parse LLM response into Q&A pairs."""
        # This is a simplified parser - in practice you'd need more robust parsing
        # based on the actual LLM response format
        try:
            # Try to parse as JSON first
            if response.strip().startswith('['):
                return json.loads(response)
            elif response.strip().startswith('{'):
                data = json.loads(response)
                return [data] if isinstance(data, dict) else data
            else:
                # Fallback: extract questions and answers using regex
                return self._extract_qa_pairs_regex(response)
        except json.JSONDecodeError:
            return self._extract_qa_pairs_regex(response)

    def _extract_qa_pairs_regex(self, response: str) -> List[dict]:
        """Extract Q&A pairs using regex as fallback."""
        import re

        # Simple regex patterns (would need to be more sophisticated)
        question_pattern = r'Question:?\s*(.*?)(?=Answer:|$)'
        answer_pattern = r'Answer:?\s*(.*?)(?=Question:|$)'

        questions = re.findall(question_pattern, response, re.IGNORECASE | re.DOTALL)
        answers = re.findall(answer_pattern, response, re.IGNORECASE | re.DOTALL)

        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.append({
                "question": q.strip(),
                "answer": a.strip(),
                "context": None,
                "metadata": {}
            })

        return qa_pairs

    def save_evaluation_data(self, qa_pairs: List[QAPair], output_file: str = None):
        """Save generated Q&A pairs to file."""
        if output_file is None:
            output_file = os.path.join(self.config.output_folder, "evaluation_data.json")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        data = {
            "metadata": {
                "total_pairs": len(qa_pairs),
                "config": self.config.__dict__,
                "generated_at": str(uuid.uuid4())
            },
            "qa_pairs": [pair.to_dict() for pair in qa_pairs]
        }

        save_json_file(output_file, data)
        print(f"Saved {len(qa_pairs)} Q&A pairs to {output_file}")
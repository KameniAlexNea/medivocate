"""Prediction generation for evaluation."""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from tqdm import tqdm

from ..config import EvaluationConfig
from ..models.evaluation_data import QAPair
from ..utils.file_utils import find_files_by_pattern, load_json_file, save_text_file
from ..utils.llm_utils import get_rag_system


class Predictor:
    """Generate predictions for evaluation questions."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.rag_system = get_rag_system()

    def generate_predictions(self, qa_pairs: List[QAPair] = None) -> List[dict]:
        """Generate predictions for Q&A pairs."""
        if qa_pairs is None:
            qa_pairs = self._load_evaluation_data()

        os.makedirs(self.config.predictions_folder, exist_ok=True)

        predictions = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._generate_single_prediction, qa_pair)
                for qa_pair in qa_pairs
            ]

            for future in tqdm(futures, desc="Generating predictions"):
                try:
                    result = future.result()
                    predictions.append(result)
                except Exception as e:
                    print(f"Error generating prediction: {e}")
                    continue

        return predictions

    def _load_evaluation_data(self) -> List[QAPair]:
        """Load evaluation data from files."""
        eval_files = find_files_by_pattern(self.config.clear_evaluation_folder, "*.json")

        qa_pairs = []
        for file_path in eval_files:
            try:
                data = load_json_file(file_path)
                qa_pair = QAPair.from_dict(data)
                qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return qa_pairs

    def _generate_single_prediction(self, qa_pair: QAPair) -> dict:
        """Generate prediction for a single Q&A pair."""
        try:
            # Generate answer using RAG system
            predicted_answer = "".join(self.rag_system.query(qa_pair.question))

            # Save prediction to file
            filename = f"{qa_pair.metadata.get('generated_id', 'unknown')}.txt"
            filepath = os.path.join(self.config.predictions_folder, filename)
            save_text_file(filepath, predicted_answer)

            return {
                "qa_pair": qa_pair,
                "predicted_answer": predicted_answer,
                "prediction_file": filepath
            }

        except Exception as e:
            print(f"Error generating prediction for question '{qa_pair.question[:50]}...': {e}")
            return {
                "qa_pair": qa_pair,
                "predicted_answer": "",
                "error": str(e)
            }

    def load_existing_predictions(self) -> List[dict]:
        """Load existing predictions from files."""
        prediction_files = find_files_by_pattern(self.config.predictions_folder, "*.txt")

        predictions = []
        for file_path in prediction_files:
            try:
                filename = os.path.basename(file_path)
                qa_file = os.path.join(
                    self.config.clear_evaluation_folder,
                    filename.replace('.txt', '.json')
                )

                if os.path.exists(qa_file):
                    qa_data = load_json_file(qa_file)
                    qa_pair = QAPair.from_dict(qa_data)
                    predicted_answer = open(file_path, 'r', encoding='utf-8').read()

                    predictions.append({
                        "qa_pair": qa_pair,
                        "predicted_answer": predicted_answer,
                        "prediction_file": file_path
                    })

            except Exception as e:
                print(f"Error loading prediction {file_path}: {e}")
                continue

        return predictions
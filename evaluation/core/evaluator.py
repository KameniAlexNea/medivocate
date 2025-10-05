"""Evaluation of predictions against ground truth."""
import os
from typing import List

from tqdm import tqdm

from ..config import EvaluationConfig
from ..models.evaluation_data import EvaluationResult, EvaluationMetrics
from ..utils.file_utils import find_files_by_pattern, load_json_file, save_json_file
from ..utils.llm_utils import get_llm_client
from ..prompts import ESCI_VALIDATOR, VALIDATOR_PROMPT_FR_CONTENT


class Evaluator:
    """Evaluate predictions against ground truth answers."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm = get_llm_client(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    def evaluate_predictions(self) -> List[EvaluationResult]:
        """Evaluate all predictions."""
        predictions = self._load_predictions()
        results = []

        for prediction in tqdm(predictions, desc="Evaluating predictions"):
            try:
                result = self._evaluate_single_prediction(prediction)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating prediction: {e}")
                continue

        return results

    def _load_predictions(self) -> List[dict]:
        """Load predictions with their corresponding Q&A pairs."""
        prediction_files = find_files_by_pattern(self.config.predictions_folder, "*.txt")

        predictions = []
        for pred_file in prediction_files:
            try:
                # Find corresponding Q&A file
                base_name = os.path.basename(pred_file).replace('.txt', '.json')
                qa_file = os.path.join(self.config.clear_evaluation_folder, base_name)

                if not os.path.exists(qa_file):
                    print(f"Warning: No Q&A file found for {pred_file}")
                    continue

                qa_data = load_json_file(qa_file)
                qa_pair = qa_data  # Already a dict, will be converted to QAPair later

                with open(pred_file, 'r', encoding='utf-8') as f:
                    predicted_answer = f.read().strip()

                predictions.append({
                    "qa_data": qa_data,
                    "predicted_answer": predicted_answer,
                    "prediction_file": pred_file
                })

            except Exception as e:
                print(f"Error loading prediction {pred_file}: {e}")
                continue

        return predictions

    def _evaluate_single_prediction(self, prediction: dict) -> EvaluationResult:
        """Evaluate a single prediction."""
        qa_data = prediction["qa_data"]
        predicted_answer = prediction["predicted_answer"]

        # Prepare evaluation prompt
        eval_prompt = self._prepare_evaluation_prompt(qa_data, predicted_answer)

        # Get evaluation from LLM
        response = self.llm.invoke([
            ("system", ESCI_VALIDATOR),
            ("user", eval_prompt)
        ])

        evaluation_text = response.content.strip()

        # Parse evaluation and score
        evaluation, score = self._parse_evaluation(evaluation_text)

        return EvaluationResult(
            question=qa_data["question"],
            expected_answer=qa_data["answer"],
            predicted_answer=predicted_answer,
            evaluation=evaluation,
            score=score,
            metadata={
                "source_file": prediction["prediction_file"],
                **qa_data.get("metadata", {})
            }
        )

    def _prepare_evaluation_prompt(self, qa_data: dict, predicted_answer: str) -> str:
        """Prepare the evaluation prompt."""
        return VALIDATOR_PROMPT_FR_CONTENT.format(
            question=qa_data["question"],
            expected_answer=qa_data["answer"],
            predicted_answer=predicted_answer,
            context=qa_data.get("context", "")
        )

    def _parse_evaluation(self, evaluation_text: str) -> tuple[str, float]:
        """Parse evaluation text to extract score and assessment."""
        # Simple parsing logic - in practice this would be more sophisticated
        text_lower = evaluation_text.lower()

        # Extract score if present
        score = None
        if "score:" in text_lower or "note:" in text_lower:
            import re
            score_match = re.search(r'(?:score|note):\s*(\d+(?:\.\d+)?)', text_lower)
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 scale if needed
                if score > 1:
                    score = score / 10.0

        # Determine evaluation category
        if "excellent" in text_lower or (score and score >= 0.9):
            evaluation = "excellent"
        elif "good" in text_lower or (score and score >= 0.7):
            evaluation = "good"
        elif "acceptable" in text_lower or (score and score >= 0.5):
            evaluation = "acceptable"
        elif "poor" in text_lower or (score and score >= 0.3):
            evaluation = "poor"
        else:
            evaluation = "bad"

        return evaluation, score

    def calculate_metrics(self, results: List[EvaluationResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics from results."""
        metrics = EvaluationMetrics()

        for result in results:
            metrics.add_result(result)

        return metrics

    def save_results(self, results: List[EvaluationResult], metrics: EvaluationMetrics):
        """Save evaluation results and metrics."""
        os.makedirs(self.config.results_folder, exist_ok=True)

        # Save detailed results
        results_file = os.path.join(self.config.results_folder, "evaluation_results.json")
        results_data = {
            "metadata": {
                "total_results": len(results),
                "config": self.config.__dict__
            },
            "results": [result.to_dict() for result in results],
            "metrics": metrics.to_dict()
        }
        save_json_file(results_file, results_data)

        # Save summary
        summary_file = os.path.join(self.config.results_folder, "evaluation_summary.json")
        save_json_file(summary_file, metrics.to_dict())

        print(f"Evaluation complete. Results saved to {self.config.results_folder}")
        print(f"Accuracy: {metrics.accuracy:.2f}%")
        print(f"Average Score: {metrics.average_score:.2f}")
        print(f"Total Questions: {metrics.total_questions}")
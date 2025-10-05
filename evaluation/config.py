"""Configuration management for the evaluation system."""
import os
from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""

    # Data paths
    input_folder: str = "data/chunks"
    output_folder: str = "data/evaluation"
    predictions_folder: str = "data/llm_eval_predictions"
    results_folder: str = "data/evaluation_results"
    clear_evaluation_folder: str = "data/clear_evaluation"

    # Processing parameters
    n_files: int = 250
    file_type: str = "json"
    max_workers: int = 2

    # LLM parameters
    temperature: float = 0.1
    max_tokens: int = 1000

    # File filtering
    min_words_threshold: int = 100

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Create config from environment variables."""
        return cls(
            input_folder=os.getenv("EVAL_INPUT_FOLDER", "data/chunks"),
            output_folder=os.getenv("EVAL_OUTPUT_FOLDER", "data/evaluation"),
            predictions_folder=os.getenv("EVAL_PREDICTIONS_FOLDER", "data/llm_eval_predictions"),
            results_folder=os.getenv("EVAL_RESULTS_FOLDER", "data/evaluation_results"),
            clear_evaluation_folder=os.getenv("EVAL_CLEAR_FOLDER", "data/clear_evaluation"),
            n_files=int(os.getenv("EVAL_N_FILES", "250")),
            file_type=os.getenv("EVAL_FILE_TYPE", "json"),
            max_workers=int(os.getenv("EVAL_MAX_WORKERS", "2")),
            temperature=float(os.getenv("EVAL_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "1000")),
            min_words_threshold=int(os.getenv("EVAL_MIN_WORDS", "100")),
        )
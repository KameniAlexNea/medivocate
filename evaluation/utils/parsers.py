"""Parsing utilities for evaluation data."""
import json
import re
from typing import Dict, List, Tuple, Any

from .file_utils import load_json_file, load_text_file


def parse_evaluation_result(raw_evaluation: str) -> Dict[str, Any]:
    """Parse raw evaluation text into structured data."""
    try:
        # Remove markdown code blocks and clean the text
        cleaned = raw_evaluation.replace("```", "").replace("json", "").strip()

        # Try to parse as JSON
        result = json.loads(cleaned)

        # Ensure required fields exist
        if "evaluation" not in result:
            result["evaluation"] = "unknown"

        return result
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: return basic structure
        return {
            "evaluation": "parsing_error",
            "error": str(e),
            "raw": raw_evaluation
        }


def extract_score_from_evaluation(evaluation_text: str) -> str:
    """Extract score/category from evaluation text."""
    # Look for common evaluation terms
    text_lower = evaluation_text.lower()

    if any(word in text_lower for word in ["excellent", "parfait", "excellent"]):
        return "excellent"
    elif any(word in text_lower for word in ["good", "bon", "bien"]):
        return "good"
    elif any(word in text_lower for word in ["acceptable", "correct", "passable"]):
        return "acceptable"
    elif any(word in text_lower for word in ["poor", "mauvais", "insuffisant"]):
        return "poor"
    else:
        return "bad"


def validate_qa_pair(question: str, answer: str) -> bool:
    """Validate that a Q&A pair is properly formed."""
    if not question or not answer:
        return False

    if len(question.strip()) < 10:
        return False

    if len(answer.strip()) < 10:
        return False

    return True


def merge_evaluation_data(query_data: Dict, prediction: str, evaluation: str) -> Dict:
    """Merge query data, prediction, and evaluation into a single structure."""
    return {
        "question": query_data.get("question", ""),
        "answer": query_data.get("answer", ""),
        "context": query_data.get("context"),
        "suggested": prediction,
        "evaluation": evaluation,
        "metadata": query_data.get("metadata", {}),
    }
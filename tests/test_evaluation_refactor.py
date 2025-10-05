"""Basic tests for the refactored evaluation package."""
import pytest
import tempfile
import os
from pathlib import Path

from evaluation.config import EvaluationConfig
from evaluation.core.metrics import EvaluationMetrics
from evaluation.models.evaluation_data import QAPair, EvaluationResult


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.input_folder == "data/chunks"
        assert config.output_folder == "data/evaluation"
        assert config.temperature == 0.1
        assert config.max_tokens == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            input_folder="custom/input",
            temperature=0.5,
            max_tokens=500
        )
        assert config.input_folder == "custom/input"
        assert config.temperature == 0.5
        assert config.max_tokens == 500


class TestEvaluationMetrics:
    """Test metrics calculation."""

    def test_empty_metrics(self):
        """Test metrics with no results."""
        metrics = EvaluationMetrics()
        metrics.calculate_metrics()
        assert metrics.total_questions == 0
        assert metrics.accuracy == 0.0

    def test_single_result(self):
        """Test metrics with a single result."""
        metrics = EvaluationMetrics()
        result = EvaluationResult(
            question="Test question?",
            expected_answer="Test answer",
            predicted_answer="Test prediction",
            evaluation="good",
            score=0.8
        )
        metrics.add_result(result)
        metrics.calculate_metrics()

        assert metrics.total_questions == 1
        assert metrics.correct_answers == 1  # score >= 0.7
        assert metrics.accuracy == 100.0
        assert metrics.average_score == 0.8

    def test_multiple_results(self):
        """Test metrics with multiple results."""
        metrics = EvaluationMetrics()

        # Add results with different scores
        results = [
            EvaluationResult("Q1", "A1", "P1", "excellent", 0.9),
            EvaluationResult("Q2", "A2", "P2", "good", 0.8),
            EvaluationResult("Q3", "A3", "P3", "poor", 0.5),
        ]

        for result in results:
            metrics.add_result(result)

        metrics.calculate_metrics()

        assert metrics.total_questions == 3
        assert metrics.correct_answers == 2  # Two scores >= 0.7
        assert metrics.accuracy == pytest.approx(66.67, rel=1e-2)
        assert metrics.average_score == pytest.approx(0.733, rel=1e-2)


class TestQAPair:
    """Test Q&A pair data model."""

    def test_qa_pair_creation(self):
        """Test creating a Q&A pair."""
        qa = QAPair(
            question="What is AI?",
            answer="Artificial Intelligence",
            context="AI refers to artificial intelligence.",
            metadata={"source": "test"}
        )

        assert qa.question == "What is AI?"
        assert qa.answer == "Artificial Intelligence"
        assert qa.context == "AI refers to artificial intelligence."
        assert qa.metadata == {"source": "test"}

    def test_qa_pair_to_dict(self):
        """Test converting Q&A pair to dictionary."""
        qa = QAPair(
            question="Test?",
            answer="Answer",
            context="Context",
            metadata={"key": "value"}
        )

        data = qa.to_dict()
        assert data["question"] == "Test?"
        assert data["answer"] == "Answer"
        assert data["context"] == "Context"
        assert data["metadata"] == {"key": "value"}


class TestEvaluationResult:
    """Test evaluation result data model."""

    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            question="Test question?",
            expected_answer="Expected",
            predicted_answer="Predicted",
            evaluation="good",
            score=0.8,
            metadata={"test": True}
        )

        assert result.question == "Test question?"
        assert result.expected_answer == "Expected"
        assert result.predicted_answer == "Predicted"
        assert result.evaluation == "good"
        assert result.score == 0.8
        assert result.metadata == {"test": True}

    def test_evaluation_result_to_dict(self):
        """Test converting evaluation result to dictionary."""
        result = EvaluationResult(
            question="Q?",
            expected_answer="A",
            predicted_answer="P",
            evaluation="excellent",
            score=0.9
        )

        data = result.to_dict()
        assert data["question"] == "Q?"
        assert data["expected_answer"] == "A"
        assert data["predicted_answer"] == "P"
        assert data["evaluation"] == "excellent"
        assert data["score"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
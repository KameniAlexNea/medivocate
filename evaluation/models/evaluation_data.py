"""Data models for evaluation system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class EvaluationScore(Enum):
    """Evaluation score categories."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    BAD = "bad"


@dataclass
class QAPair:
    """Question-Answer pair data structure."""

    question: str
    answer: str
    context: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QAPair":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            context=data.get("context"),
            metadata=data.get("metadata", {}),
        )

    def __post_init__(self):
        """Validate data after initialization."""
        if not self.question.strip():
            raise ValueError("Question cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")


@dataclass
class EvaluationResult:
    """Evaluation result data structure."""

    question: str
    expected_answer: str
    predicted_answer: str
    evaluation: str
    score: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.expected_answer,
            "suggested": self.predicted_answer,
            "evaluation": self.evaluation,
            "score": self.score,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            expected_answer=data["answer"],
            predicted_answer=data["suggested"],
            evaluation=data["evaluation"],
            score=data.get("score"),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_correct(self) -> bool:
        """Check if the prediction is considered correct."""
        if self.score is not None:
            return self.score >= 0.7  # Threshold for correctness
        return (
            "excellent" in self.evaluation.lower() or "good" in self.evaluation.lower()
        )


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total_questions: int = 0
    correct_answers: int = 0
    average_score: float = 0.0
    score_distribution: Dict[str, int] = None

    def __post_init__(self):
        if self.score_distribution is None:
            self.score_distribution = {}

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        return (
            (self.correct_answers / self.total_questions) * 100
            if self.total_questions > 0
            else 0.0
        )

    def add_result(self, result: EvaluationResult):
        """Add a result to the metrics."""
        self.total_questions += 1
        if result.is_correct:
            self.correct_answers += 1

        # Update score distribution
        score_category = self._categorize_score(result.score)
        self.score_distribution[score_category] = (
            self.score_distribution.get(score_category, 0) + 1
        )

        # Recalculate average
        self._update_average()

    def _categorize_score(self, score: Optional[float]) -> str:
        """Categorize score into buckets."""
        if score is None:
            return "unscored"
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        elif score >= 0.3:
            return "poor"
        else:
            return "bad"

    def _update_average(self):
        """Update average score (simplified calculation)."""
        # This is a simplified calculation - in practice you'd track all scores
        self.average_score = self.accuracy / 100.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": self.accuracy,
            "average_score": self.average_score,
            "score_distribution": self.score_distribution,
        }

    score: Optional[EvaluationScore] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.expected_answer,
            "suggested": self.predicted_answer,
            "evaluation": self.evaluation,
            "score": self.score.value if self.score else None,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            expected_answer=data["answer"],
            predicted_answer=data["suggested"],
            evaluation=data["evaluation"],
            score=EvaluationScore(data["score"]) if data.get("score") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    total_evaluations: int = 0
    score_distribution: Dict[EvaluationScore, int] = None
    average_score: Optional[float] = None

    def __post_init__(self):
        if self.score_distribution is None:
            self.score_distribution = {score: 0 for score in EvaluationScore}

    def add_result(self, result: EvaluationResult):
        """Add a result to the metrics."""
        self.total_evaluations += 1
        if result.score:
            self.score_distribution[result.score] += 1

    def calculate_average(self):
        """Calculate average score based on distribution."""
        if self.total_evaluations == 0:
            return 0.0

        # Assign numerical values to scores
        score_values = {
            EvaluationScore.EXCELLENT: 5,
            EvaluationScore.GOOD: 4,
            EvaluationScore.ACCEPTABLE: 3,
            EvaluationScore.POOR: 2,
            EvaluationScore.BAD: 1,
        }

        total_score = sum(
            count * score_values[score]
            for score, count in self.score_distribution.items()
        )
        self.average_score = total_score / self.total_evaluations
        return self.average_score

    def get_percentage_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of scores."""
        if self.total_evaluations == 0:
            return {score.value: 0.0 for score in EvaluationScore}

        return {
            score.value: (count / self.total_evaluations) * 100
            for score, count in self.score_distribution.items()
        }

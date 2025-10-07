"""Metrics calculation for evaluation results."""

from collections import Counter
from typing import Any, Dict


class EvaluationMetrics:
    """Aggregate metrics for evaluation results."""

    def __init__(self):
        self.total_questions = 0
        self.correct_answers = 0
        self.accuracy = 0.0
        self.average_score = 0.0
        self.score_distribution = Counter()
        self.evaluation_distribution = Counter()
        self.scores = []

    def add_result(self, result):
        """Add a single evaluation result to metrics."""
        self.total_questions += 1
        self.scores.append(result.score)

        # Count evaluation categories
        self.evaluation_distribution[result.evaluation] += 1

        # Count score ranges
        if result.score is not None:
            score_range = self._get_score_range(result.score)
            self.score_distribution[score_range] += 1

            # Consider answer correct if score >= 0.7
            if result.score >= 0.7:
                self.correct_answers += 1

    def _get_score_range(self, score: float) -> str:
        """Get score range category."""
        if score >= 0.9:
            return "excellent (0.9-1.0)"
        elif score >= 0.8:
            return "very_good (0.8-0.9)"
        elif score >= 0.7:
            return "good (0.7-0.8)"
        elif score >= 0.6:
            return "acceptable (0.6-0.7)"
        elif score >= 0.5:
            return "poor (0.5-0.6)"
        else:
            return "bad (0.0-0.5)"

    def calculate_metrics(self):
        """Calculate aggregate metrics."""
        if self.total_questions == 0:
            return

        self.accuracy = (self.correct_answers / self.total_questions) * 100
        self.average_score = sum(
            score for score in self.scores if score is not None
        ) / len([s for s in self.scores if s is not None])

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self.calculate_metrics()

        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "accuracy": round(self.accuracy, 2),
            "average_score": (
                round(self.average_score, 3) if self.average_score else None
            ),
            "score_distribution": dict(self.score_distribution),
            "evaluation_distribution": dict(self.evaluation_distribution),
            "score_percentiles": self._calculate_percentiles(),
        }

    def _calculate_percentiles(self) -> Dict[str, float]:
        """Calculate score percentiles."""
        if not self.scores:
            return {}

        valid_scores = [s for s in self.scores if s is not None]
        if not valid_scores:
            return {}

        valid_scores.sort()

        def percentile(p):
            k = (len(valid_scores) - 1) * (p / 100)
            f = int(k)
            c = k - f
            if f + 1 < len(valid_scores):
                return round(
                    valid_scores[f] + c * (valid_scores[f + 1] - valid_scores[f]), 3
                )
            else:
                return round(valid_scores[f], 3)

        return {
            "25th_percentile": percentile(25),
            "50th_percentile": percentile(50),
            "75th_percentile": percentile(75),
            "90th_percentile": percentile(90),
            "95th_percentile": percentile(95),
        }

    def print_summary(self):
        """Print a summary of the metrics."""
        self.calculate_metrics()

        print("\n" + "=" * 50)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 50)
        print(f"Total Questions: {self.total_questions}")
        print(f"Correct Answers: {self.correct_answers}")
        print(f"Accuracy: {self.accuracy:.2f}%")
        print(
            f"Average Score: {self.average_score:.3f}"
            if self.average_score
            else "Average Score: N/A"
        )

        print("\nScore Distribution:")
        for category, count in sorted(self.score_distribution.items()):
            percentage = (count / self.total_questions) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        print("\nEvaluation Distribution:")
        for category, count in sorted(self.evaluation_distribution.items()):
            percentage = (count / self.total_questions) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        percentiles = self._calculate_percentiles()
        if percentiles:
            print("\nScore Percentiles:")
            for p, value in percentiles.items():
                print(f"  {p}: {value}")

        print("=" * 50)

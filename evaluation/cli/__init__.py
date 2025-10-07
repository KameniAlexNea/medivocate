"""Command-line interface for evaluation package."""

import argparse
import sys
from pathlib import Path

from ..config import EvaluationConfig
from ..core.data_generator import DataGenerator
from ..core.evaluator import Evaluator
from ..core.predictor import Predictor


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Medivocate Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate evaluation data
  python -m evaluation.cli generate --num-questions 100 --output-dir data/evaluation

  # Run predictions
  python -m evaluation.cli predict --input-dir data/evaluation --output-dir data/predictions

  # Evaluate predictions
  python -m evaluation.cli evaluate --predictions-dir data/predictions --results-dir data/results

  # Run full pipeline
  python -m evaluation.cli pipeline --num-questions 50 --output-dir data/evaluation
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate evaluation data")
    generate_parser.add_argument(
        "--num-questions", type=int, default=100, help="Number of Q&A pairs to generate"
    )
    generate_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated data",
    )
    generate_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size for text processing"
    )
    generate_parser.add_argument(
        "--overlap", type=int, default=200, help="Overlap between chunks"
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature for generation"
    )
    generate_parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum tokens for LLM responses"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Run predictions on evaluation data"
    )
    predict_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing evaluation data",
    )
    predict_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for predictions"
    )
    predict_parser.add_argument(
        "--temperature", type=float, default=0.1, help="LLM temperature for predictions"
    )
    predict_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Maximum tokens for predictions"
    )
    predict_parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of worker threads"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    evaluate_parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing predictions",
    )
    evaluate_parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    evaluate_parser.add_argument(
        "--temperature", type=float, default=0.1, help="LLM temperature for evaluation"
    )
    evaluate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for evaluation responses",
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run full evaluation pipeline"
    )
    pipeline_parser.add_argument(
        "--num-questions", type=int, default=50, help="Number of Q&A pairs to generate"
    )
    pipeline_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for all pipeline steps",
    )
    pipeline_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Chunk size for text processing"
    )
    pipeline_parser.add_argument(
        "--overlap", type=int, default=200, help="Overlap between chunks"
    )
    pipeline_parser.add_argument(
        "--gen-temperature",
        type=float,
        default=0.7,
        help="LLM temperature for data generation",
    )
    pipeline_parser.add_argument(
        "--pred-temperature",
        type=float,
        default=0.1,
        help="LLM temperature for predictions",
    )
    pipeline_parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.1,
        help="LLM temperature for evaluation",
    )
    pipeline_parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of worker threads"
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "generate":
            run_generate(args)
        elif args.command == "predict":
            run_predict(args)
        elif args.command == "evaluate":
            run_evaluate(args)
        elif args.command == "pipeline":
            run_pipeline(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_generate(args):
    """Run data generation."""
    config = EvaluationConfig(
        clear_evaluation_folder=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    generator = DataGenerator(config)
    generator.generate_evaluation_data(args.num_questions)

    print(f"Generated {args.num_questions} Q&A pairs in {args.output_dir}")


def run_predict(args):
    """Run predictions."""
    config = EvaluationConfig(
        clear_evaluation_folder=args.input_dir,
        predictions_folder=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
    )

    predictor = Predictor(config)
    predictor.run_predictions()

    print(f"Predictions completed. Results saved to {args.output_dir}")


def run_evaluate(args):
    """Run evaluation."""
    config = EvaluationConfig(
        predictions_folder=args.predictions_dir,
        results_folder=args.results_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate_predictions()
    metrics = evaluator.calculate_metrics(results)
    evaluator.save_results(results, metrics)

    metrics.print_summary()


def run_pipeline(args):
    """Run full evaluation pipeline."""
    base_dir = Path(args.output_dir)
    eval_dir = base_dir / "evaluation_data"
    pred_dir = base_dir / "predictions"
    results_dir = base_dir / "results"

    # Step 1: Generate data
    print("Step 1: Generating evaluation data...")
    gen_config = EvaluationConfig(
        clear_evaluation_folder=str(eval_dir),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        temperature=args.gen_temperature,
        max_tokens=1000,
    )
    generator = DataGenerator(gen_config)
    generator.generate_evaluation_data(args.num_questions)

    # Step 2: Run predictions
    print("\nStep 2: Running predictions...")
    pred_config = EvaluationConfig(
        clear_evaluation_folder=str(eval_dir),
        predictions_folder=str(pred_dir),
        temperature=args.pred_temperature,
        max_tokens=500,
        max_workers=args.max_workers,
    )
    predictor = Predictor(pred_config)
    predictor.run_predictions()

    # Step 3: Evaluate
    print("\nStep 3: Evaluating predictions...")
    eval_config = EvaluationConfig(
        predictions_folder=str(pred_dir),
        results_folder=str(results_dir),
        temperature=args.eval_temperature,
        max_tokens=1000,
    )
    evaluator = Evaluator(eval_config)
    results = evaluator.evaluate_predictions()
    metrics = evaluator.calculate_metrics(results)
    evaluator.save_results(results, metrics)

    print("\nPipeline completed successfully!")
    metrics.print_summary()


if __name__ == "__main__":
    main()

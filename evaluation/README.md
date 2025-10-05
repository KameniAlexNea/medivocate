# Medivocate Evaluation Package

A comprehensive evaluation system for testing and benchmarking RAG (Retrieval-Augmented Generation) systems.

## Features

- **Automated Q&A Generation**: Generate evaluation datasets from documents using LLM-powered question generation
- **Prediction Pipeline**: Run predictions on evaluation datasets with configurable concurrency
- **Comprehensive Evaluation**: Evaluate predictions against ground truth with detailed metrics
- **CLI Interface**: Command-line tools for all evaluation steps
- **Pipeline Automation**: Run complete evaluation pipelines with a single command

## Installation

The evaluation package is part of the medivocate project. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### Full Pipeline Evaluation

Run a complete evaluation pipeline with 50 Q&A pairs:

```bash
python -m evaluation.cli pipeline --num-questions 50 --output-dir evaluation_results
```

This will:
1. Generate evaluation data from your documents
2. Run predictions using your RAG system
3. Evaluate the predictions and generate metrics

### Individual Steps

#### 1. Generate Evaluation Data

```bash
python -m evaluation.cli generate \
  --num-questions 100 \
  --output-dir data/evaluation \
  --chunk-size 1000 \
  --overlap 200
```

#### 2. Run Predictions

```bash
python -m evaluation.cli predict \
  --input-dir data/evaluation \
  --output-dir data/predictions \
  --max-workers 4
```

#### 3. Evaluate Results

```bash
python -m evaluation.cli evaluate \
  --predictions-dir data/predictions \
  --results-dir data/results
```

## Configuration

The evaluation system uses a centralized configuration system. Key parameters:

- **LLM Settings**: Temperature, max tokens, model selection
- **Text Processing**: Chunk size, overlap for document processing
- **Concurrency**: Number of worker threads for parallel processing
- **Output Directories**: Configurable paths for all outputs

## Output Structure

```
evaluation_results/
├── evaluation_data/     # Generated Q&A pairs (JSON)
├── predictions/         # Model predictions (TXT files)
└── results/            # Evaluation results and metrics
    ├── evaluation_results.json
    └── evaluation_summary.json
```

## Metrics

The evaluation system provides comprehensive metrics:

- **Accuracy**: Percentage of correct answers (score ≥ 0.7)
- **Average Score**: Mean evaluation score across all predictions
- **Score Distribution**: Breakdown by score ranges
- **Evaluation Categories**: Distribution of qualitative assessments
- **Percentiles**: 25th, 50th, 75th, 90th, and 95th percentiles

## API Usage

```python
from evaluation import EvaluationConfig, DataGenerator, Predictor, Evaluator

# Configure evaluation
config = EvaluationConfig(
    clear_evaluation_folder="data/evaluation",
    predictions_folder="data/predictions",
    results_folder="data/results"
)

# Generate data
generator = DataGenerator(config)
generator.generate_evaluation_data(num_questions=100)

# Run predictions
predictor = Predictor(config)
predictor.run_predictions()

# Evaluate results
evaluator = Evaluator(config)
results = evaluator.evaluate_predictions()
metrics = evaluator.calculate_metrics(results)
evaluator.save_results(results, metrics)
```

## Architecture

The evaluation package is organized into clear modules:

- **`core/`**: Core evaluation logic
  - `data_generator.py`: Q&A pair generation
  - `predictor.py`: Prediction pipeline
  - `evaluator.py`: Result evaluation
  - `metrics.py`: Metrics calculation

- **`models/`**: Data models and schemas
- **`utils/`**: Utility functions
- **`cli/`**: Command-line interface

## Dependencies

- LangChain for LLM integration
- tqdm for progress bars
- concurrent.futures for parallel processing
- pathlib for file operations

## Contributing

When adding new evaluation methods:

1. Add the evaluation logic to the appropriate core module
2. Update the CLI interface if needed
3. Add comprehensive tests
4. Update this README with new features

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**: Check your API keys and network connectivity
2. **Memory Issues**: Reduce chunk size or number of workers
3. **File Not Found**: Ensure input directories exist and contain expected files

### Debug Mode

Enable verbose logging by setting the log level in your configuration.
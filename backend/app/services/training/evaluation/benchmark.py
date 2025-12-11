"""
Benchmarking framework for comparing bioinformatics code models.

Supports:
- Multi-model comparison
- Standardized test suites
- Performance tracking over time
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import Dataset

from .evaluator import EvaluationResult, ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    benchmark_name: str
    run_at: datetime
    models: list[str]

    # Per-model results
    results: dict[str, EvaluationResult] = field(default_factory=dict)

    # Comparison metrics
    rankings: dict[str, list[str]] = field(default_factory=dict)  # metric -> [model ranking]
    best_model: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "run_at": self.run_at.isoformat(),
            "models": self.models,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "rankings": self.rankings,
            "best_model": self.best_model,
        }


class BenchmarkRunner:
    """
    Runs benchmarks comparing multiple models.
    """

    # Standard benchmark tasks for bioinformatics
    STANDARD_TASKS = [
        {
            "name": "sequence_analysis",
            "description": "Analyze DNA/RNA sequences",
            "languages": ["python", "r"],
            "packages": ["biopython", "Biostrings"],
        },
        {
            "name": "differential_expression",
            "description": "Perform differential expression analysis",
            "languages": ["r", "python"],
            "packages": ["DESeq2", "edgeR", "pydeseq2"],
        },
        {
            "name": "visualization",
            "description": "Create bioinformatics visualizations",
            "languages": ["r", "python"],
            "packages": ["ComplexHeatmap", "ggplot2", "matplotlib", "seaborn"],
        },
        {
            "name": "single_cell",
            "description": "Single-cell RNA-seq analysis",
            "languages": ["python", "r"],
            "packages": ["scanpy", "Seurat", "scater"],
        },
        {
            "name": "alignment",
            "description": "Sequence alignment tasks",
            "languages": ["python"],
            "packages": ["pysam", "biopython"],
        },
    ]

    def __init__(
        self,
        output_dir: str = "benchmarks",
        execute_code: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execute_code = execute_code

    def run_benchmark(
        self,
        models: dict[str, dict],  # name -> {path, base_model (optional)}
        test_dataset: Dataset,
        benchmark_name: str = "bioinfo_benchmark",
    ) -> BenchmarkResult:
        """
        Run benchmark on multiple models.

        Args:
            models: Dictionary mapping model names to their configs
            test_dataset: Test dataset to evaluate on
            benchmark_name: Name for this benchmark

        Returns:
            BenchmarkResult with comparison
        """
        logger.info(f"Running benchmark: {benchmark_name}")
        logger.info(f"Models: {list(models.keys())}")
        logger.info(f"Test examples: {len(test_dataset)}")

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            run_at=datetime.now(),
            models=list(models.keys()),
        )

        # Evaluate each model
        for model_name, model_config in models.items():
            logger.info(f"Evaluating model: {model_name}")

            evaluator = ModelEvaluator()
            evaluator.load_model(
                model_path=model_config["path"],
                base_model=model_config.get("base_model"),
            )

            eval_result = evaluator.evaluate(
                dataset=test_dataset,
                dataset_name=benchmark_name,
                execute_code=self.execute_code,
            )

            result.results[model_name] = eval_result

        # Compute rankings
        result.rankings = self._compute_rankings(result.results)

        # Determine best model
        result.best_model = self._determine_best(result.results)

        # Save results
        self._save_benchmark(result)

        return result

    def run_quick_benchmark(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        num_examples: int = 50,
    ) -> EvaluationResult:
        """
        Run a quick benchmark on a single model.

        Args:
            model_path: Path to model
            base_model: Base model for LoRA
            num_examples: Number of examples to test

        Returns:
            EvaluationResult
        """
        from .test_dataset import create_test_dataset

        logger.info("Running quick benchmark")

        # Create test dataset
        test_dataset = create_test_dataset(num_examples=num_examples)

        # Evaluate
        evaluator = ModelEvaluator()
        evaluator.load_model(model_path, base_model)

        result = evaluator.evaluate(
            dataset=test_dataset,
            dataset_name="quick_benchmark",
            max_examples=num_examples,
        )

        return result

    def compare_to_baseline(
        self,
        model_path: str,
        baseline_model: str,
        test_dataset: Dataset,
        base_model: Optional[str] = None,
    ) -> dict:
        """
        Compare a fine-tuned model to its baseline.

        Args:
            model_path: Path to fine-tuned model
            baseline_model: Name of baseline model
            test_dataset: Test dataset
            base_model: Base model for LoRA

        Returns:
            Comparison results
        """
        models = {
            "baseline": {"path": baseline_model},
            "fine_tuned": {"path": model_path, "base_model": base_model},
        }

        result = self.run_benchmark(
            models=models,
            test_dataset=test_dataset,
            benchmark_name="baseline_comparison",
        )

        baseline_score = result.results["baseline"].avg_score
        finetuned_score = result.results["fine_tuned"].avg_score

        return {
            "baseline_score": baseline_score,
            "fine_tuned_score": finetuned_score,
            "improvement": finetuned_score - baseline_score,
            "improvement_percent": (
                (finetuned_score - baseline_score) / max(baseline_score, 0.001) * 100
            ),
            "rankings": result.rankings,
        }

    def _compute_rankings(
        self,
        results: dict[str, EvaluationResult],
    ) -> dict[str, list[str]]:
        """Compute rankings for each metric."""
        rankings = {}

        # Metrics to rank on
        metrics = [
            ("avg_score", True),
            ("syntax_valid_rate", True),
            ("execution_success_rate", True),
        ]

        for metric, higher_is_better in metrics:
            scores = []
            for model_name, result in results.items():
                score = getattr(result, metric, 0.0)
                scores.append((model_name, score))

            scores.sort(key=lambda x: x[1], reverse=higher_is_better)
            rankings[metric] = [name for name, _ in scores]

        return rankings

    def _determine_best(
        self,
        results: dict[str, EvaluationResult],
    ) -> str:
        """Determine the best model based on overall score."""
        best_model = None
        best_score = -1

        for model_name, result in results.items():
            if result.avg_score > best_score:
                best_score = result.avg_score
                best_model = model_name

        return best_model

    def _save_benchmark(self, result: BenchmarkResult) -> Path:
        """Save benchmark results."""
        filename = f"{result.benchmark_name}_{result.run_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved benchmark to {filepath}")
        return filepath

    def load_benchmark(self, filepath: str) -> dict:
        """Load a previous benchmark result."""
        with open(filepath) as f:
            return json.load(f)

    def get_benchmark_history(
        self,
        benchmark_name: Optional[str] = None,
    ) -> list[dict]:
        """Get history of benchmark runs."""
        history = []

        for file in self.output_dir.glob("*.json"):
            data = self.load_benchmark(str(file))
            if benchmark_name is None or data.get("benchmark_name") == benchmark_name:
                history.append({
                    "file": str(file),
                    "benchmark_name": data.get("benchmark_name"),
                    "run_at": data.get("run_at"),
                    "models": data.get("models", []),
                    "best_model": data.get("best_model"),
                })

        history.sort(key=lambda x: x["run_at"], reverse=True)
        return history


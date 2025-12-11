"""
Evaluation framework for bioinformatics code models.

Provides:
- Model evaluation on test datasets
- Code quality metrics
- Benchmarking against baseline models
"""

from .evaluator import ModelEvaluator, EvaluationResult
from .metrics import CodeMetrics, compute_code_metrics
from .benchmark import BenchmarkRunner, BenchmarkResult
from .test_dataset import BioinfoTestDataset, create_test_dataset

__all__ = [
    "ModelEvaluator",
    "EvaluationResult",
    "CodeMetrics",
    "compute_code_metrics",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BioinfoTestDataset",
    "create_test_dataset",
]


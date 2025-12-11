"""
Training services for fine-tuning bioinformatics coding models.

This package provides:
- Data collection from GitHub, documentation, and paper repositories
- Data preprocessing for instruction-following format
- Dataset building with Hugging Face datasets library
- Training with LoRA/QLoRA support
- Model evaluation and benchmarking
"""

from .data_collector import (
    CodeExample,
    DataCollector,
    GitHubCollector,
    DocumentationCollector,
    BioconductorCollector,
    BiopythonCollector,
)
from .data_preprocessor import DataPreprocessor, TrainingExample
from .dataset_builder import DatasetBuilder, create_training_dataset
from .data_validator import DataValidator, ValidationResult
from .training_config import TrainingConfig, LoRAConfig, TrainingSettings, training_settings
from .model_loader import ModelLoader
from .trainer import BioinfoCodeTrainer, TrainingJob, TrainingStatus

__all__ = [
    # Data collection
    "CodeExample",
    "DataCollector",
    "GitHubCollector",
    "DocumentationCollector",
    "BioconductorCollector",
    "BiopythonCollector",
    # Preprocessing
    "DataPreprocessor",
    "TrainingExample",
    # Dataset building
    "DatasetBuilder",
    "create_training_dataset",
    # Validation
    "DataValidator",
    "ValidationResult",
    # Configuration
    "TrainingConfig",
    "LoRAConfig",
    "TrainingSettings",
    "training_settings",
    # Model loading
    "ModelLoader",
    # Training
    "BioinfoCodeTrainer",
    "TrainingJob",
    "TrainingStatus",
]


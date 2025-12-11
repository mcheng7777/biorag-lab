"""
Dataset builder for creating Hugging Face datasets from preprocessed examples.

Creates datasets in formats suitable for:
- Supervised fine-tuning (SFT)
- Instruction tuning
- Chat/conversational format
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Value

from .data_preprocessor import TrainingExample

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds Hugging Face datasets from preprocessed training examples.

    Supports multiple output formats:
    - Standard instruction format (instruction, input, output)
    - Chat format (messages with roles)
    - Alpaca format (for compatibility)
    """

    def __init__(self, output_dir: str = "data/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_instruction_dataset(
        self,
        examples: list[TrainingExample],
        dataset_name: str = "bioinfo_code",
    ) -> Dataset:
        """
        Build a dataset in standard instruction format.

        Format:
        {
            "instruction": str,
            "input": str,
            "output": str,
            "language": str,
            "source": str,
        }
        """
        data = []
        for example in examples:
            data.append({
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
                "language": example.language,
                "source": example.source,
                "example_id": example.example_id,
            })

        features = Features({
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
            "language": Value("string"),
            "source": Value("string"),
            "example_id": Value("string"),
        })

        dataset = Dataset.from_list(data, features=features)
        logger.info(f"Built instruction dataset with {len(dataset)} examples")

        return dataset

    def build_chat_dataset(
        self,
        examples: list[TrainingExample],
        dataset_name: str = "bioinfo_code_chat",
    ) -> Dataset:
        """
        Build a dataset in chat/conversational format.

        Format:
        {
            "messages": [
                {"role": "system", "content": str},
                {"role": "user", "content": str},
                {"role": "assistant", "content": str},
            ]
        }
        """
        data = []
        for example in examples:
            messages = example.to_chat_format()
            data.append({
                "messages": messages,
                "language": example.language,
                "source": example.source,
                "example_id": example.example_id,
            })

        dataset = Dataset.from_list(data)
        logger.info(f"Built chat dataset with {len(dataset)} examples")

        return dataset

    def build_alpaca_dataset(
        self,
        examples: list[TrainingExample],
        dataset_name: str = "bioinfo_code_alpaca",
    ) -> Dataset:
        """
        Build a dataset in Alpaca format for compatibility.

        Format:
        {
            "instruction": str,
            "input": str,
            "output": str,
        }
        """
        data = []
        for example in examples:
            data.append({
                "instruction": example.instruction,
                "input": example.input,
                "output": example.output,
            })

        features = Features({
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
        })

        dataset = Dataset.from_list(data, features=features)
        logger.info(f"Built Alpaca dataset with {len(dataset)} examples")

        return dataset

    def build_dataset_dict(
        self,
        train: list[TrainingExample],
        validation: list[TrainingExample],
        test: Optional[list[TrainingExample]] = None,
        format: str = "instruction",
    ) -> DatasetDict:
        """
        Build a DatasetDict with train/validation/test splits.

        Args:
            train: Training examples
            validation: Validation examples
            test: Test examples (optional)
            format: Dataset format ("instruction", "chat", or "alpaca")

        Returns:
            DatasetDict with splits
        """
        build_fn = {
            "instruction": self.build_instruction_dataset,
            "chat": self.build_chat_dataset,
            "alpaca": self.build_alpaca_dataset,
        }.get(format, self.build_instruction_dataset)

        splits = {
            "train": build_fn(train),
            "validation": build_fn(validation),
        }

        if test:
            splits["test"] = build_fn(test)

        dataset_dict = DatasetDict(splits)
        logger.info(
            f"Built DatasetDict: train={len(train)}, "
            f"validation={len(validation)}, "
            f"test={len(test) if test else 0}"
        )

        return dataset_dict

    def save_dataset(
        self,
        dataset: Dataset | DatasetDict,
        name: str,
        push_to_hub: bool = False,
        hub_repo: Optional[str] = None,
    ) -> Path:
        """
        Save dataset to disk and optionally push to Hugging Face Hub.

        Args:
            dataset: Dataset or DatasetDict to save
            name: Name for the dataset
            push_to_hub: Whether to push to Hugging Face Hub
            hub_repo: Repository name for Hub (required if push_to_hub=True)

        Returns:
            Path to saved dataset
        """
        save_path = self.output_dir / name

        # Save to disk
        dataset.save_to_disk(str(save_path))
        logger.info(f"Saved dataset to {save_path}")

        # Also save as JSONL for compatibility
        jsonl_path = self.output_dir / f"{name}.jsonl"
        self._save_as_jsonl(dataset, jsonl_path)

        # Push to Hub if requested
        if push_to_hub:
            if not hub_repo:
                raise ValueError("hub_repo required when push_to_hub=True")
            dataset.push_to_hub(hub_repo)
            logger.info(f"Pushed dataset to Hub: {hub_repo}")

        return save_path

    def _save_as_jsonl(
        self,
        dataset: Dataset | DatasetDict,
        path: Path,
    ) -> None:
        """Save dataset as JSONL file(s)."""
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_path = path.parent / f"{path.stem}_{split_name}.jsonl"
                self._save_split_as_jsonl(split_dataset, split_path)
        else:
            self._save_split_as_jsonl(dataset, path)

    def _save_split_as_jsonl(self, dataset: Dataset, path: Path) -> None:
        """Save a single dataset split as JSONL."""
        with open(path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved JSONL to {path}")

    def load_dataset(self, name: str) -> DatasetDict | Dataset:
        """Load a dataset from disk."""
        from datasets import load_from_disk

        load_path = self.output_dir / name
        dataset = load_from_disk(str(load_path))
        logger.info(f"Loaded dataset from {load_path}")
        return dataset

    def get_dataset_stats(
        self,
        dataset: Dataset | DatasetDict,
    ) -> dict:
        """Get statistics about a dataset."""
        stats = {}

        if isinstance(dataset, DatasetDict):
            stats["splits"] = {}
            for split_name, split_dataset in dataset.items():
                stats["splits"][split_name] = self._get_split_stats(split_dataset)
            stats["total_examples"] = sum(
                s["num_examples"] for s in stats["splits"].values()
            )
        else:
            stats = self._get_split_stats(dataset)

        return stats

    def _get_split_stats(self, dataset: Dataset) -> dict:
        """Get statistics for a single dataset split."""
        stats = {
            "num_examples": len(dataset),
            "columns": dataset.column_names,
        }

        # Calculate length statistics if possible
        if "output" in dataset.column_names:
            output_lengths = [len(ex["output"]) for ex in dataset]
            stats["output_length"] = {
                "min": min(output_lengths),
                "max": max(output_lengths),
                "mean": sum(output_lengths) / len(output_lengths),
            }

        if "instruction" in dataset.column_names:
            instruction_lengths = [len(ex["instruction"]) for ex in dataset]
            stats["instruction_length"] = {
                "min": min(instruction_lengths),
                "max": max(instruction_lengths),
                "mean": sum(instruction_lengths) / len(instruction_lengths),
            }

        # Language distribution
        if "language" in dataset.column_names:
            languages = [ex["language"] for ex in dataset]
            stats["language_distribution"] = {
                lang: languages.count(lang) for lang in set(languages)
            }

        return stats


def create_training_dataset(
    examples: list[TrainingExample],
    output_dir: str = "data/datasets",
    dataset_name: str = "bioinfo_code",
    format: str = "instruction",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
) -> tuple[DatasetDict, Path]:
    """
    Convenience function to create a complete training dataset.

    Args:
        examples: List of TrainingExample objects
        output_dir: Directory to save the dataset
        dataset_name: Name for the dataset
        format: Dataset format ("instruction", "chat", or "alpaca")
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        test_ratio: Fraction for test split
        push_to_hub: Whether to push to Hugging Face Hub
        hub_repo: Repository name for Hub

    Returns:
        Tuple of (DatasetDict, save_path)
    """
    from .data_preprocessor import DataPreprocessor

    # Split the examples
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.split_dataset(
        examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # Build dataset
    builder = DatasetBuilder(output_dir=output_dir)
    dataset_dict = builder.build_dataset_dict(
        train=train,
        validation=val,
        test=test,
        format=format,
    )

    # Save
    save_path = builder.save_dataset(
        dataset_dict,
        name=dataset_name,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )

    # Log stats
    stats = builder.get_dataset_stats(dataset_dict)
    logger.info(f"Dataset stats: {stats}")

    return dataset_dict, save_path


"""
Main training script for fine-tuning bioinformatics coding models.

Provides:
- SFTTrainer integration
- LoRA/QLoRA training
- Training job management
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import TrainingArguments

from .callbacks import get_default_callbacks
from .model_loader import ModelLoader
from .training_config import TrainingConfig, training_settings

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a training job."""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Optional[dict] = None
    output_dir: Optional[str] = None


class BioinfoCodeTrainer:
    """
    Trainer for fine-tuning coding models on bioinformatics data.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or training_settings.default_config
        self.model_loader = ModelLoader(
            cache_dir=training_settings.cache_dir,
        )
        self._trainer = None
        self._current_job: Optional[TrainingJob] = None

    def prepare_dataset(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: Optional[int] = None,
    ) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            dataset: Raw dataset with instruction/input/output columns
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Tokenized dataset
        """
        max_length = max_length or self.config.max_seq_length

        def format_example(example):
            """Format a single example for training."""
            # Build the full text
            text = f"### Instruction:\n{example['instruction']}\n\n"

            if example.get("input"):
                text += f"### Input:\n{example['input']}\n\n"

            text += f"### Response:\n{example['output']}"

            return {"text": text}

        # Format examples
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
        )

        return formatted_dataset

    def create_training_args(self) -> TrainingArguments:
        """Create training arguments from config."""
        args_dict = self.config.get_training_args_dict()

        # Ensure output directory exists
        output_dir = Path(args_dict["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        return TrainingArguments(**args_dict)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingJob:
        """
        Run training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            TrainingJob with results
        """
        from trl import SFTTrainer

        # Create job
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = TrainingJob(
            job_id=job_id,
            config=self.config,
            status=TrainingStatus.PENDING,
            created_at=datetime.now(),
            output_dir=str(Path(self.config.output_dir) / job_id),
        )
        self._current_job = job

        try:
            logger.info(f"Starting training job: {job_id}")
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()

            # Load model and tokenizer
            model, tokenizer = self.model_loader.load_for_training(self.config)

            # Prepare datasets
            train_formatted = self.prepare_dataset(train_dataset, tokenizer)
            eval_formatted = None
            if eval_dataset is not None:
                eval_formatted = self.prepare_dataset(eval_dataset, tokenizer)

            # Create training arguments
            training_args = self.create_training_args()
            training_args.output_dir = job.output_dir

            # Get callbacks
            callbacks = get_default_callbacks(
                output_dir=job.output_dir,
                wandb_project=self.config.wandb_project,
            )

            # Create trainer
            self._trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_formatted,
                eval_dataset=eval_formatted,
                tokenizer=tokenizer,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                callbacks=callbacks,
            )

            # Train
            train_result = self._trainer.train(
                resume_from_checkpoint=resume_from_checkpoint,
            )

            # Save final model
            self._trainer.save_model()
            tokenizer.save_pretrained(job.output_dir)

            # Update job
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.metrics = {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples_per_second": train_result.metrics.get(
                    "train_samples_per_second"
                ),
            }

            # Run final evaluation if eval dataset provided
            if eval_formatted is not None:
                eval_result = self._trainer.evaluate()
                job.metrics["eval_loss"] = eval_result.get("eval_loss")

            logger.info(f"Training completed: {job_id}")
            logger.info(f"Final metrics: {job.metrics}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            raise

        finally:
            self._current_job = job

        return job

    def evaluate(
        self,
        eval_dataset: Dataset,
    ) -> dict:
        """
        Evaluate the current model on a dataset.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Evaluation metrics
        """
        if self._trainer is None:
            raise ValueError("No trainer available. Train a model first.")

        _, tokenizer = self.model_loader.load_for_training(self.config)
        eval_formatted = self.prepare_dataset(eval_dataset, tokenizer)

        return self._trainer.evaluate(eval_formatted)

    def get_current_job(self) -> Optional[TrainingJob]:
        """Get the current or last training job."""
        return self._current_job

    def cancel_training(self) -> bool:
        """Cancel the current training job."""
        if self._current_job and self._current_job.status == TrainingStatus.RUNNING:
            self._current_job.status = TrainingStatus.CANCELLED
            self._current_job.completed_at = datetime.now()
            logger.info(f"Training cancelled: {self._current_job.job_id}")
            return True
        return False


def train_bioinformatics_model(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    output_dir: str = "models/fine_tuned",
) -> TrainingJob:
    """
    Convenience function to train a bioinformatics coding model.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
        output_dir: Output directory for model

    Returns:
        TrainingJob with results
    """
    if config is None:
        config = training_settings.default_config.model_copy()
        config.output_dir = output_dir

    trainer = BioinfoCodeTrainer(config=config)
    return trainer.train(train_dataset, eval_dataset)


async def train_async(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainingJob:
    """
    Async wrapper for training (runs training in executor).

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration

    Returns:
        TrainingJob with results
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        job = await loop.run_in_executor(
            executor,
            lambda: train_bioinformatics_model(train_dataset, eval_dataset, config),
        )

    return job


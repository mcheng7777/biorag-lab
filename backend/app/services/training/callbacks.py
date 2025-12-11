"""
Training callbacks for monitoring and controlling the fine-tuning process.

Provides:
- Weights & Biases integration
- Checkpointing
- Early stopping
- Resource monitoring
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """
    Callback for enhanced logging during training.
    """

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.start_time = time.time()
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info(f"  Total steps: {state.max_steps}")
        logger.info(f"  Epochs: {args.num_train_epochs}")
        logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info("=" * 60)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.log_every_n_steps == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            eta = (state.max_steps - state.global_step) / steps_per_sec if steps_per_sec > 0 else 0

            logger.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"Loss: {state.log_history[-1].get('loss', 0):.4f} | "
                f"Speed: {steps_per_sec:.2f} steps/s | "
                f"ETA: {eta / 60:.1f} min"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info("Training completed")
        logger.info(f"  Total time: {elapsed / 60:.1f} minutes")
        logger.info(f"  Final loss: {state.log_history[-1].get('loss', 'N/A')}")
        logger.info("=" * 60)


class GPUMemoryCallback(TrainerCallback):
    """
    Callback for monitoring GPU memory usage.
    """

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % self.log_every_n_steps == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3

                logger.info(
                    f"GPU Memory - Allocated: {allocated:.2f}GB | "
                    f"Reserved: {reserved:.2f}GB | "
                    f"Max: {max_allocated:.2f}GB"
                )


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping based on evaluation loss.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.patience_counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.patience_counter = 0
            logger.info(f"New best eval loss: {eval_loss:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement in eval loss. "
                f"Patience: {self.patience_counter}/{self.patience}"
            )

            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered!")
                control.should_training_stop = True


class SaveBestModelCallback(TrainerCallback):
    """
    Save the best model based on evaluation metric.
    """

    def __init__(
        self,
        output_dir: str,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = float("-inf") if greater_is_better else float("inf")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        if metrics is None or model is None:
            return

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return

        is_better = (
            current_metric > self.best_metric
            if self.greater_is_better
            else current_metric < self.best_metric
        )

        if is_better:
            self.best_metric = current_metric
            best_path = self.output_dir / "best_model"
            best_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model.save_pretrained(str(best_path))
            logger.info(
                f"New best model saved! {self.metric_name}: {current_metric:.4f}"
            )


class WandbCallback(TrainerCallback):
    """
    Enhanced Weights & Biases logging callback.
    """

    def __init__(
        self,
        project: str = "bioinfo-code-finetuning",
        run_name: Optional[str] = None,
        log_model: bool = True,
    ):
        self.project = project
        self.run_name = run_name
        self.log_model = log_model
        self._wandb = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        try:
            import wandb

            self._wandb = wandb

            if wandb.run is None:
                wandb.init(
                    project=self.project,
                    name=self.run_name or args.run_name,
                    config={
                        "learning_rate": args.learning_rate,
                        "epochs": args.num_train_epochs,
                        "batch_size": args.per_device_train_batch_size,
                        "gradient_accumulation": args.gradient_accumulation_steps,
                        "max_steps": state.max_steps,
                    },
                )

            # Log model architecture
            if model is not None:
                trainable = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in model.parameters())
                wandb.log({
                    "model/trainable_params": trainable,
                    "model/total_params": total,
                    "model/trainable_percent": 100 * trainable / total,
                })

        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if self._wandb is not None and logs is not None:
            # Add GPU memory stats
            if torch.cuda.is_available():
                logs["gpu/memory_allocated_gb"] = (
                    torch.cuda.memory_allocated() / 1024**3
                )
                logs["gpu/memory_reserved_gb"] = (
                    torch.cuda.memory_reserved() / 1024**3
                )

            self._wandb.log(logs, step=state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if self._wandb is not None:
            if self.log_model and model is not None:
                # Save model artifact
                artifact = self._wandb.Artifact(
                    name=f"model-{self._wandb.run.id}",
                    type="model",
                )
                artifact.add_dir(args.output_dir)
                self._wandb.log_artifact(artifact)

            self._wandb.finish()


def get_default_callbacks(
    output_dir: str,
    wandb_project: Optional[str] = None,
    early_stopping_patience: int = 3,
) -> list[TrainerCallback]:
    """
    Get a list of default callbacks for training.

    Args:
        output_dir: Directory to save models
        wandb_project: W&B project name (None to disable)
        early_stopping_patience: Patience for early stopping

    Returns:
        List of TrainerCallback instances
    """
    callbacks = [
        LoggingCallback(log_every_n_steps=10),
        GPUMemoryCallback(log_every_n_steps=50),
        EarlyStoppingCallback(patience=early_stopping_patience),
        SaveBestModelCallback(output_dir=output_dir),
    ]

    if wandb_project and os.environ.get("WANDB_API_KEY"):
        callbacks.append(WandbCallback(project=wandb_project))

    return callbacks


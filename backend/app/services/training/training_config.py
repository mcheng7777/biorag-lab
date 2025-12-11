"""
Training configuration for fine-tuning bioinformatics coding models.

Provides configuration classes for:
- LoRA/QLoRA parameters
- Training hyperparameters
- Model selection
- Resource management
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelType(str, Enum):
    """Supported base models for fine-tuning."""

    CODELLAMA_7B = "codellama/CodeLlama-7b-hf"
    CODELLAMA_13B = "codellama/CodeLlama-13b-hf"
    STARCODER2_7B = "bigcode/starcoder2-7b"
    STARCODER2_15B = "bigcode/starcoder2-15b"
    DEEPSEEK_CODER_7B = "deepseek-ai/deepseek-coder-6.7b-base"
    DEEPSEEK_CODER_33B = "deepseek-ai/deepseek-coder-33b-base"
    # The model referenced in existing config
    GPT_OSS_20B = "openai/gpt-oss-20b"


class QuantizationType(str, Enum):
    """Quantization options for memory-efficient training."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # QLoRA's normalized float 4-bit


class LoRAConfig(BaseModel):
    """Configuration for LoRA adapters."""

    # LoRA rank
    r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA attention dimension (rank)",
    )

    # LoRA alpha
    lora_alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA scaling factor",
    )

    # LoRA dropout
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers",
    )

    # Target modules to apply LoRA
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Module names to apply LoRA to",
    )

    # Bias handling
    bias: str = Field(
        default="none",
        description="Bias type for LoRA: 'none', 'all', or 'lora_only'",
    )

    # Task type
    task_type: str = Field(
        default="CAUSAL_LM",
        description="Task type for PEFT",
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters configuration."""

    # Model selection
    base_model: str = Field(
        default=ModelType.CODELLAMA_7B.value,
        description="Base model to fine-tune",
    )

    # Output settings
    output_dir: str = Field(
        default="models/fine_tuned",
        description="Directory to save fine-tuned models",
    )
    run_name: str = Field(
        default="bioinfo_code_ft",
        description="Name for this training run",
    )

    # Training parameters
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of training epochs",
    )
    per_device_train_batch_size: int = Field(
        default=4,
        ge=1,
        description="Training batch size per device",
    )
    per_device_eval_batch_size: int = Field(
        default=4,
        ge=1,
        description="Evaluation batch size per device",
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Gradient accumulation steps",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0,
        description="Learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay",
    )
    warmup_ratio: float = Field(
        default=0.03,
        ge=0,
        le=1,
        description="Warmup ratio of total steps",
    )
    max_grad_norm: float = Field(
        default=0.3,
        ge=0,
        description="Maximum gradient norm for clipping",
    )

    # Sequence length
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum sequence length",
    )

    # Optimizer
    optim: str = Field(
        default="paged_adamw_32bit",
        description="Optimizer to use",
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        description="Learning rate scheduler type",
    )

    # Precision and memory
    bf16: bool = Field(
        default=True,
        description="Use bfloat16 precision",
    )
    fp16: bool = Field(
        default=False,
        description="Use float16 precision",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Enable gradient checkpointing",
    )

    # Quantization
    quantization: QuantizationType = Field(
        default=QuantizationType.NF4,
        description="Quantization type for QLoRA",
    )

    # LoRA configuration
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA adapter configuration",
    )

    # Logging and saving
    logging_steps: int = Field(
        default=10,
        ge=1,
        description="Log every N steps",
    )
    save_steps: int = Field(
        default=100,
        ge=1,
        description="Save checkpoint every N steps",
    )
    eval_steps: int = Field(
        default=100,
        ge=1,
        description="Evaluate every N steps",
    )
    save_total_limit: int = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep",
    )

    # Evaluation
    evaluation_strategy: str = Field(
        default="steps",
        description="Evaluation strategy: 'steps', 'epoch', or 'no'",
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="Load best model at end of training",
    )
    metric_for_best_model: str = Field(
        default="eval_loss",
        description="Metric to use for best model selection",
    )

    # Weights & Biases
    report_to: str = Field(
        default="wandb",
        description="Where to report training metrics",
    )
    wandb_project: str = Field(
        default="bioinfo-code-finetuning",
        description="W&B project name",
    )

    # Hardware
    dataloader_num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of dataloader workers",
    )

    def get_training_args_dict(self) -> dict:
        """Convert to dictionary for TrainingArguments."""
        return {
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "evaluation_strategy": self.evaluation_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "report_to": self.report_to,
            "dataloader_num_workers": self.dataloader_num_workers,
        }


class TrainingSettings(BaseSettings):
    """Global training settings from environment."""

    # Paths
    data_dir: str = Field(
        default="data/training",
        description="Directory for training data",
    )
    models_dir: str = Field(
        default="models",
        description="Directory for models",
    )
    cache_dir: str = Field(
        default=".cache/huggingface",
        description="Hugging Face cache directory",
    )

    # Hugging Face
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face API token",
    )
    hub_model_id: Optional[str] = Field(
        default=None,
        description="Hub repository for pushing models",
    )

    # Weights & Biases
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="W&B API key",
    )
    wandb_entity: Optional[str] = Field(
        default=None,
        description="W&B entity/team name",
    )

    # Resource limits
    max_gpu_memory_gb: float = Field(
        default=16.0,
        ge=1.0,
        description="Maximum GPU memory to use",
    )

    # Default training config
    default_config: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Default training configuration",
    )

    class Config:
        env_prefix = "TRAINING_"
        env_file = ".env"

    def get_output_path(self, run_name: str) -> Path:
        """Get output path for a training run."""
        path = Path(self.models_dir) / "fine_tuned" / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_checkpoint_path(self, run_name: str, step: int) -> Path:
        """Get checkpoint path for a specific step."""
        return self.get_output_path(run_name) / f"checkpoint-{step}"


# Create settings instances
training_settings = TrainingSettings()


def get_qlora_config() -> tuple[TrainingConfig, LoRAConfig]:
    """
    Get optimized configuration for QLoRA training on 16GB GPU.

    Returns:
        Tuple of (TrainingConfig, LoRAConfig) optimized for QLoRA
    """
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_config = TrainingConfig(
        base_model=ModelType.CODELLAMA_7B.value,
        quantization=QuantizationType.NF4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
        lora=lora_config,
    )

    return training_config, lora_config


def get_full_finetune_config() -> TrainingConfig:
    """
    Get configuration for full fine-tuning (requires more GPU memory).

    Returns:
        TrainingConfig for full fine-tuning
    """
    return TrainingConfig(
        base_model=ModelType.CODELLAMA_7B.value,
        quantization=QuantizationType.NONE,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=2048,
    )


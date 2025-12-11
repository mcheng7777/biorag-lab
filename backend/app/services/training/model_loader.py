"""
Model loading utilities for fine-tuning.

Provides:
- Base model loading with quantization
- LoRA adapter setup
- Memory-efficient loading options
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .training_config import LoRAConfig, QuantizationType, TrainingConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading base models and setting up LoRA adapters.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/huggingface",
        device_map: str = "auto",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device_map = device_map

    def get_quantization_config(
        self,
        quantization: QuantizationType,
    ) -> Optional[BitsAndBytesConfig]:
        """
        Get quantization configuration for model loading.

        Args:
            quantization: Type of quantization to apply

        Returns:
            BitsAndBytesConfig or None if no quantization
        """
        if quantization == QuantizationType.NONE:
            return None

        if quantization == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )

        if quantization in (QuantizationType.INT4, QuantizationType.NF4):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4" if quantization == QuantizationType.NF4 else "fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        return None

    def load_tokenizer(
        self,
        model_name: str,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizer:
        """
        Load tokenizer for the model.

        Args:
            model_name: Hugging Face model name or path
            trust_remote_code: Whether to trust remote code

        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=trust_remote_code,
        )

        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side for causal LM
        tokenizer.padding_side = "right"

        return tokenizer

    def load_base_model(
        self,
        model_name: str,
        config: TrainingConfig,
        trust_remote_code: bool = True,
    ) -> PreTrainedModel:
        """
        Load base model with optional quantization.

        Args:
            model_name: Hugging Face model name or path
            config: Training configuration
            trust_remote_code: Whether to trust remote code

        Returns:
            Loaded model
        """
        logger.info(f"Loading base model: {model_name}")

        # Get quantization config
        bnb_config = self.get_quantization_config(config.quantization)

        # Determine dtype
        if config.bf16:
            torch_dtype = torch.bfloat16
        elif config.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            cache_dir=str(self.cache_dir),
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2" if self._supports_flash_attention() else None,
        )

        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and bnb_config is None:
            model.gradient_checkpointing_enable()

        logger.info(f"Model loaded with {self._count_parameters(model)} parameters")

        return model

    def get_lora_config(self, lora_config: LoRAConfig) -> LoraConfig:
        """
        Convert LoRAConfig to PEFT LoraConfig.

        Args:
            lora_config: Our LoRA configuration

        Returns:
            PEFT LoraConfig
        """
        return LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAConfig,
    ) -> PeftModel:
        """
        Apply LoRA adapters to a model.

        Args:
            model: Base model to apply LoRA to
            lora_config: LoRA configuration

        Returns:
            Model with LoRA adapters
        """
        logger.info(f"Applying LoRA with r={lora_config.r}, alpha={lora_config.lora_alpha}")

        peft_config = self.get_lora_config(lora_config)
        model = get_peft_model(model, peft_config)

        # Log trainable parameters
        trainable, total = self._count_trainable_parameters(model)
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        return model

    def load_for_training(
        self,
        config: TrainingConfig,
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load model and tokenizer ready for training.

        Args:
            config: Training configuration

        Returns:
            Tuple of (model with LoRA, tokenizer)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer(config.base_model)

        # Load base model
        model = self.load_base_model(config.base_model, config)

        # Apply LoRA
        model = self.apply_lora(model, config.lora)

        return model, tokenizer

    def load_fine_tuned_model(
        self,
        base_model_name: str,
        adapter_path: str,
        quantization: QuantizationType = QuantizationType.NF4,
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load a fine-tuned model with LoRA adapters.

        Args:
            base_model_name: Name of the base model
            adapter_path: Path to saved LoRA adapters
            quantization: Quantization type for loading

        Returns:
            Tuple of (model with adapters, tokenizer)
        """
        logger.info(f"Loading fine-tuned model from {adapter_path}")

        # Create a minimal config for loading
        config = TrainingConfig(
            base_model=base_model_name,
            quantization=quantization,
        )

        # Load tokenizer
        tokenizer = self.load_tokenizer(base_model_name)

        # Load base model
        base_model = self.load_base_model(base_model_name, config)

        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )

        logger.info("Fine-tuned model loaded successfully")

        return model, tokenizer

    def save_model(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        save_full_model: bool = False,
    ) -> Path:
        """
        Save model and tokenizer.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            output_dir: Directory to save to
            save_full_model: If True, merge LoRA and save full model

        Returns:
            Path to saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_full_model:
            # Merge LoRA weights and save full model
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(output_path))
        else:
            # Save only LoRA adapters
            model.save_pretrained(str(output_path))

        tokenizer.save_pretrained(str(output_path))

        logger.info(f"Model saved to {output_path}")

        return output_path

    def _supports_flash_attention(self) -> bool:
        """Check if flash attention is available."""
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    def _count_parameters(self, model: PreTrainedModel) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in model.parameters())

    def _count_trainable_parameters(
        self,
        model: PreTrainedModel,
    ) -> tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total


def get_model_memory_footprint(model: PreTrainedModel) -> dict:
    """
    Estimate memory footprint of a model.

    Returns:
        Dictionary with memory statistics
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024,
        "total_size_mb": total_size / 1024 / 1024,
        "total_size_gb": total_size / 1024 / 1024 / 1024,
    }


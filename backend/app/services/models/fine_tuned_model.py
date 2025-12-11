"""
Fine-tuned model service for bioinformatics code generation.

Handles:
- Model loading with LoRA adapters
- Inference with configurable parameters
- Memory management and optimization
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    model_id: str
    base_model: str
    adapter_path: Optional[str]
    is_fine_tuned: bool
    loaded_at: datetime
    device: str
    dtype: str
    memory_footprint_mb: float

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "is_fine_tuned": self.is_fine_tuned,
            "loaded_at": self.loaded_at.isoformat(),
            "device": self.device,
            "dtype": self.dtype,
            "memory_footprint_mb": self.memory_footprint_mb,
        }


class FineTunedModel:
    """
    Service for loading and using fine-tuned bioinformatics code models.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/huggingface",
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = self.dtype_map.get(dtype, torch.bfloat16)

        self.model: Optional[PeftModel | AutoModelForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_info: Optional[ModelInfo] = None

    def load_base_model(
        self,
        model_name: str,
        quantization: str = "none",
    ) -> "FineTunedModel":
        """
        Load a base model without fine-tuning.

        Args:
            model_name: Hugging Face model name
            quantization: Quantization type ("none", "int8", "int4")

        Returns:
            self for chaining
        """
        logger.info(f"Loading base model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
        load_kwargs = {
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if quantization == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        else:
            load_kwargs["torch_dtype"] = self.dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        # Create model info
        self.model_info = ModelInfo(
            model_id=model_name,
            base_model=model_name,
            adapter_path=None,
            is_fine_tuned=False,
            loaded_at=datetime.now(),
            device=self.device,
            dtype=str(self.dtype),
            memory_footprint_mb=self._get_memory_footprint(),
        )

        logger.info(f"Base model loaded: {self.model_info.memory_footprint_mb:.1f}MB")
        return self

    def load_fine_tuned(
        self,
        base_model: str,
        adapter_path: str,
        quantization: str = "int4",
    ) -> "FineTunedModel":
        """
        Load a fine-tuned model with LoRA adapters.

        Args:
            base_model: Base model name
            adapter_path: Path to LoRA adapter weights
            quantization: Quantization type

        Returns:
            self for chaining
        """
        logger.info(f"Loading fine-tuned model: {base_model} + {adapter_path}")

        # First load base model
        self.load_base_model(base_model, quantization)

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )
        self.model.eval()

        # Update model info
        self.model_info = ModelInfo(
            model_id=f"{base_model}+{Path(adapter_path).name}",
            base_model=base_model,
            adapter_path=adapter_path,
            is_fine_tuned=True,
            loaded_at=datetime.now(),
            device=self.device,
            dtype=str(self.dtype),
            memory_footprint_mb=self._get_memory_footprint(),
        )

        logger.info(f"Fine-tuned model loaded: {self.model_info.memory_footprint_mb:.1f}MB")
        return self

    def generate(
        self,
        instruction: str,
        context: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Generate code for an instruction.

        Args:
            instruction: The instruction/prompt
            context: Additional context (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated code string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded. Call load_base_model or load_fine_tuned first.")

        # Format prompt
        prompt = self._format_prompt(instruction, context)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated.strip()

    def generate_batch(
        self,
        instructions: list[str],
        contexts: Optional[list[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        """
        Generate code for multiple instructions.

        Args:
            instructions: List of instructions
            contexts: List of contexts (optional)
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            List of generated code strings
        """
        if contexts is None:
            contexts = [""] * len(instructions)

        results = []
        for instruction, context in zip(instructions, contexts):
            result = self.generate(
                instruction=instruction,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(result)

        return results

    def _format_prompt(self, instruction: str, context: str = "") -> str:
        """Format instruction and context into a prompt."""
        prompt = f"### Instruction:\n{instruction}\n\n"
        if context:
            prompt += f"### Input:\n{context}\n\n"
        prompt += "### Response:\n"
        return prompt

    def _get_memory_footprint(self) -> float:
        """Get model memory footprint in MB."""
        if self.model is None:
            return 0.0

        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())

        return (param_size + buffer_size) / 1024 / 1024

    def get_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded model."""
        return self.model_info

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None

    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.model_info = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


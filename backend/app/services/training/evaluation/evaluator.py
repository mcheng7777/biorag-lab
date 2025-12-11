"""
Model evaluator for bioinformatics code generation.

Evaluates models on:
- Test dataset performance
- Code quality metrics
- Task-specific accuracy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from tqdm import tqdm

from .metrics import CodeMetrics, aggregate_metrics, compute_code_metrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    model_name: str
    dataset_name: str
    evaluated_at: datetime
    num_examples: int

    # Aggregate metrics
    avg_score: float
    syntax_valid_rate: float
    execution_success_rate: float

    # Detailed metrics
    metrics_by_language: dict = field(default_factory=dict)
    metrics_by_task: dict = field(default_factory=dict)

    # Per-example results
    example_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "evaluated_at": self.evaluated_at.isoformat(),
            "num_examples": self.num_examples,
            "avg_score": self.avg_score,
            "syntax_valid_rate": self.syntax_valid_rate,
            "execution_success_rate": self.execution_success_rate,
            "metrics_by_language": self.metrics_by_language,
            "metrics_by_task": self.metrics_by_task,
        }


class ModelEvaluator:
    """
    Evaluates code generation models on bioinformatics tasks.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "auto",
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def load_model(
        self,
        model_path: str,
        base_model: Optional[str] = None,
    ):
        """
        Load a model for evaluation.

        Args:
            model_path: Path to model or LoRA adapters
            base_model: Base model name (required for LoRA)
        """
        from ..model_loader import ModelLoader
        from ..training_config import QuantizationType

        loader = ModelLoader()

        if base_model:
            # Load as LoRA model
            self.model, self.tokenizer = loader.load_fine_tuned_model(
                base_model_name=base_model,
                adapter_path=model_path,
                quantization=QuantizationType.NF4,
            )
        else:
            # Load as full model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.model.eval()
        logger.info(f"Loaded model from {model_path}")

    def generate(
        self,
        instruction: str,
        context: str = "",
        language: str = "python",
    ) -> str:
        """
        Generate code for an instruction.

        Args:
            instruction: The instruction/prompt
            context: Additional context
            language: Target language

        Returns:
            Generated code
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        # Format prompt
        prompt = f"### Instruction:\n{instruction}\n\n"
        if context:
            prompt += f"### Input:\n{context}\n\n"
        prompt += "### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated.strip()

    def evaluate(
        self,
        dataset: Dataset,
        dataset_name: str = "test",
        execute_code: bool = False,
        max_examples: Optional[int] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset with instruction/input/output columns
            dataset_name: Name for this evaluation
            execute_code: Whether to attempt code execution
            max_examples: Maximum examples to evaluate (None for all)
            show_progress: Show progress bar

        Returns:
            EvaluationResult with metrics
        """
        logger.info(f"Evaluating on {dataset_name} ({len(dataset)} examples)")

        # Limit examples if specified
        if max_examples and max_examples < len(dataset):
            dataset = dataset.select(range(max_examples))

        all_metrics = []
        example_results = []
        metrics_by_language = {"python": [], "r": []}

        iterator = tqdm(dataset, desc="Evaluating") if show_progress else dataset

        for example in iterator:
            instruction = example.get("instruction", "")
            context = example.get("input", "")
            expected = example.get("output", "")
            language = example.get("language", "python")

            # Generate code
            try:
                generated = self.generate(instruction, context, language)
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                generated = ""

            # Compute metrics
            metrics = compute_code_metrics(
                generated,
                language,
                execute_code=execute_code,
            )

            all_metrics.append(metrics)
            if language in metrics_by_language:
                metrics_by_language[language].append(metrics)

            # Store example result
            example_results.append({
                "instruction": instruction[:100],
                "generated": generated[:500],
                "expected": expected[:500] if expected else None,
                "language": language,
                "metrics": metrics.to_dict(),
            })

        # Aggregate results
        aggregated = aggregate_metrics(all_metrics)

        # Aggregate by language
        lang_aggregated = {}
        for lang, metrics_list in metrics_by_language.items():
            if metrics_list:
                lang_aggregated[lang] = aggregate_metrics(metrics_list)

        result = EvaluationResult(
            model_name=self._get_model_name(),
            dataset_name=dataset_name,
            evaluated_at=datetime.now(),
            num_examples=len(dataset),
            avg_score=aggregated.get("avg_overall_score", 0.0),
            syntax_valid_rate=aggregated.get("syntax_valid_rate", 0.0),
            execution_success_rate=aggregated.get("execution_success_rate", 0.0),
            metrics_by_language=lang_aggregated,
            example_results=example_results,
        )

        logger.info(f"Evaluation complete: score={result.avg_score:.3f}")

        return result

    def compare_outputs(
        self,
        generated: str,
        expected: str,
        language: str,
    ) -> dict:
        """
        Compare generated code with expected code.

        Args:
            generated: Generated code
            expected: Expected/reference code
            language: Programming language

        Returns:
            Comparison metrics
        """
        gen_metrics = compute_code_metrics(generated, language)
        exp_metrics = compute_code_metrics(expected, language)

        # Check if same packages are used
        gen_packages = set(gen_metrics.detected_packages)
        exp_packages = set(exp_metrics.detected_packages)
        package_overlap = len(gen_packages & exp_packages) / max(len(exp_packages), 1)

        # Check structural similarity
        structure_match = (
            gen_metrics.has_functions == exp_metrics.has_functions
            and gen_metrics.has_imports == exp_metrics.has_imports
        )

        return {
            "generated_valid": gen_metrics.syntax_valid,
            "expected_valid": exp_metrics.syntax_valid,
            "package_overlap": package_overlap,
            "structure_match": structure_match,
            "generated_score": gen_metrics.overall_score,
            "expected_score": exp_metrics.overall_score,
        }

    def _get_model_name(self) -> str:
        """Get model name for reporting."""
        if hasattr(self.model, "name_or_path"):
            return self.model.name_or_path
        if hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path"):
            return self.model.config._name_or_path
        return "unknown"

    def save_results(
        self,
        result: EvaluationResult,
        output_dir: str = "evaluations",
    ) -> Path:
        """Save evaluation results to disk."""
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{result.model_name.replace('/', '_')}_{result.dataset_name}_{result.evaluated_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Also save example results separately
        examples_path = output_path / f"{filename.replace('.json', '_examples.jsonl')}"
        with open(examples_path, "w") as f:
            for ex in result.example_results:
                f.write(json.dumps(ex) + "\n")

        logger.info(f"Saved results to {filepath}")
        return filepath


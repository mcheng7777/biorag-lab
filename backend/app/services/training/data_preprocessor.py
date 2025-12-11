"""
Data preprocessing for converting collected code examples into training format.

Converts raw code examples into instruction-following format suitable for
supervised fine-tuning of language models.
"""

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from .data_collector import CodeExample

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """
    A preprocessed training example in instruction-following format.
    """

    instruction: str
    input: str  # Context/documentation
    output: str  # Expected code
    language: str
    source: str
    example_id: str
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for dataset creation."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "language": self.language,
            "source": self.source,
            "example_id": self.example_id,
            "metadata": self.metadata,
        }

    def to_chat_format(self) -> list[dict]:
        """Convert to chat format for conversational fine-tuning."""
        messages = []

        # System message
        messages.append({
            "role": "system",
            "content": f"You are an expert bioinformatics programmer specializing in {self.language}. "
            "Generate clean, well-documented code that follows best practices."
        })

        # User message with instruction and context
        user_content = self.instruction
        if self.input:
            user_content += f"\n\nContext:\n{self.input}"

        messages.append({
            "role": "user",
            "content": user_content,
        })

        # Assistant response with code
        messages.append({
            "role": "assistant",
            "content": self.output,
        })

        return messages


class DataPreprocessor:
    """
    Preprocesses collected code examples for training.

    Handles:
    - Instruction generation from code
    - Context extraction
    - Code cleaning and normalization
    - Train/validation/test splits
    """

    # Instruction templates for different code patterns
    PYTHON_INSTRUCTION_TEMPLATES = [
        "Write Python code to {task}.",
        "Create a Python function that {task}.",
        "Implement {task} using Python.",
        "Generate Python code for {task}.",
        "Write a Python script to {task}.",
    ]

    R_INSTRUCTION_TEMPLATES = [
        "Write R code to {task}.",
        "Create an R function that {task}.",
        "Implement {task} using R.",
        "Generate R code for {task}.",
        "Write an R script to {task}.",
    ]

    # Package-specific instruction templates
    PACKAGE_TEMPLATES = {
        "biopython": [
            "Using Biopython, {task}.",
            "Write Python code with Biopython to {task}.",
            "Implement {task} using Biopython library.",
        ],
        "scanpy": [
            "Using Scanpy, {task}.",
            "Write single-cell analysis code with Scanpy to {task}.",
            "Implement {task} for single-cell RNA-seq using Scanpy.",
        ],
        "DESeq2": [
            "Using DESeq2, {task}.",
            "Write R code with DESeq2 to {task}.",
            "Implement differential expression analysis with DESeq2 to {task}.",
        ],
        "ComplexHeatmap": [
            "Using ComplexHeatmap, {task}.",
            "Create a heatmap visualization with ComplexHeatmap to {task}.",
            "Generate a heatmap using ComplexHeatmap to {task}.",
        ],
        "Seurat": [
            "Using Seurat, {task}.",
            "Write single-cell analysis code with Seurat to {task}.",
            "Implement {task} for single-cell data using Seurat.",
        ],
    }

    def __init__(
        self,
        max_code_length: int = 4096,
        max_instruction_length: int = 512,
        deduplicate: bool = True,
    ):
        self.max_code_length = max_code_length
        self.max_instruction_length = max_instruction_length
        self.deduplicate = deduplicate
        self._seen_hashes: set[str] = set()

    def preprocess(self, example: CodeExample) -> Optional[TrainingExample]:
        """
        Preprocess a single code example into training format.

        Args:
            example: Raw CodeExample from collection

        Returns:
            TrainingExample or None if preprocessing fails
        """
        # Clean the code
        cleaned_code = self._clean_code(example.code, example.language)
        if not cleaned_code:
            return None

        # Truncate if too long
        if len(cleaned_code) > self.max_code_length:
            cleaned_code = self._truncate_code(cleaned_code, example.language)

        # Check for duplicates
        if self.deduplicate:
            code_hash = self._hash_code(cleaned_code)
            if code_hash in self._seen_hashes:
                return None
            self._seen_hashes.add(code_hash)

        # Generate instruction
        instruction = self._generate_instruction(
            cleaned_code, example.language, example.packages
        )

        # Extract context
        context = self._extract_context(example)

        # Generate unique ID
        example_id = self._generate_id(example)

        return TrainingExample(
            instruction=instruction,
            input=context,
            output=cleaned_code,
            language=example.language,
            source=example.source,
            example_id=example_id,
            metadata={
                "packages": example.packages,
                "tags": example.tags,
                "source_url": example.source_url,
            },
        )

    def preprocess_batch(
        self,
        examples: list[CodeExample],
        shuffle: bool = True,
    ) -> list[TrainingExample]:
        """
        Preprocess a batch of examples.

        Args:
            examples: List of CodeExample objects
            shuffle: Whether to shuffle the output

        Returns:
            List of TrainingExample objects
        """
        training_examples = []

        for example in examples:
            try:
                processed = self.preprocess(example)
                if processed:
                    training_examples.append(processed)
            except Exception as e:
                logger.warning(f"Error preprocessing example: {e}")

        if shuffle:
            random.shuffle(training_examples)

        logger.info(
            f"Preprocessed {len(training_examples)}/{len(examples)} examples"
        )
        return training_examples

    def split_dataset(
        self,
        examples: list[TrainingExample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
        """
        Split examples into train/validation/test sets.

        Args:
            examples: List of TrainingExample objects
            train_ratio: Fraction for training (default 0.8)
            val_ratio: Fraction for validation (default 0.1)
            test_ratio: Fraction for test (default 0.1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train, validation, test) examples
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        logger.info(
            f"Split dataset: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test

    def _clean_code(self, code: str, language: str) -> Optional[str]:
        """Clean and normalize code."""
        if not code or not code.strip():
            return None

        # Remove excessive whitespace
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            cleaned_lines.append(line)

        # Remove excessive blank lines (more than 2 consecutive)
        result_lines = []
        blank_count = 0

        for line in cleaned_lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        code = "\n".join(result_lines).strip()

        # Remove common boilerplate
        code = self._remove_boilerplate(code, language)

        return code if code else None

    def _remove_boilerplate(self, code: str, language: str) -> str:
        """Remove common boilerplate code."""
        if language == "python":
            # Remove shebang
            code = re.sub(r'^#!.*\n', '', code)
            # Remove encoding declarations
            code = re.sub(r'^#.*coding[:=].*\n', '', code)
            # Remove if __name__ == '__main__' blocks at the end
            code = re.sub(
                r'\nif\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*\n.*$',
                '',
                code,
                flags=re.DOTALL
            )
        else:  # R
            # Remove common R boilerplate
            code = re.sub(r'^#!/usr/bin/env Rscript\n', '', code)

        return code.strip()

    def _truncate_code(self, code: str, language: str) -> str:
        """Truncate code to maximum length while keeping it valid."""
        lines = code.split("\n")
        truncated_lines = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > self.max_code_length:
                break
            truncated_lines.append(line)
            current_length += line_length

        # Try to end at a natural boundary
        result = "\n".join(truncated_lines)

        # For Python, try to end at a function/class boundary
        if language == "python":
            # Find the last complete function/class definition
            match = re.search(
                r'(.*\n(?:def|class)\s+\w+.*?(?=\n(?:def|class)|\Z))',
                result,
                re.DOTALL
            )
            if match:
                result = match.group(1)

        return result.strip()

    def _generate_instruction(
        self,
        code: str,
        language: str,
        packages: list[str],
    ) -> str:
        """Generate an instruction for the code."""
        # Extract task description from code
        task = self._extract_task(code, language)

        # Choose template based on packages
        templates = None
        for package in packages:
            if package in self.PACKAGE_TEMPLATES:
                templates = self.PACKAGE_TEMPLATES[package]
                break

        if templates is None:
            templates = (
                self.PYTHON_INSTRUCTION_TEMPLATES
                if language == "python"
                else self.R_INSTRUCTION_TEMPLATES
            )

        template = random.choice(templates)
        instruction = template.format(task=task)

        # Truncate if needed
        if len(instruction) > self.max_instruction_length:
            instruction = instruction[:self.max_instruction_length - 3] + "..."

        return instruction

    def _extract_task(self, code: str, language: str) -> str:
        """Extract a task description from code."""
        # Try to get from docstring (Python)
        if language == "python":
            docstring_match = re.search(r'"""([^"]+)"""', code)
            if docstring_match:
                task = docstring_match.group(1).strip().split("\n")[0]
                if len(task) > 20:
                    return task.lower()

        # Try to get from first comment
        lines = code.split("\n")
        for line in lines[:10]:
            if language == "python" and line.strip().startswith("#"):
                comment = line.strip()[1:].strip()
                if len(comment) > 20 and not comment.startswith("!"):
                    return comment.lower()
            elif language == "r" and line.strip().startswith("#"):
                comment = line.strip()[1:].strip()
                if len(comment) > 20:
                    return comment.lower()

        # Generate task from function/class names
        task = self._generate_task_from_code(code, language)
        return task

    def _generate_task_from_code(self, code: str, language: str) -> str:
        """Generate a task description from code structure."""
        tasks = []

        if language == "python":
            # Extract function names
            func_matches = re.findall(r'def\s+(\w+)\s*\(', code)
            if func_matches:
                func_name = func_matches[0]
                task = self._camel_to_words(func_name)
                tasks.append(task)

            # Extract class names
            class_matches = re.findall(r'class\s+(\w+)', code)
            if class_matches:
                class_name = class_matches[0]
                tasks.append(f"implement a {self._camel_to_words(class_name)} class")

        else:  # R
            # Extract function names
            func_matches = re.findall(r'(\w+)\s*<-\s*function\s*\(', code)
            if func_matches:
                func_name = func_matches[0]
                task = self._camel_to_words(func_name)
                tasks.append(task)

        if tasks:
            return tasks[0]

        # Fallback: generic task based on detected packages
        return "perform bioinformatics analysis"

    def _camel_to_words(self, name: str) -> str:
        """Convert camelCase or snake_case to words."""
        # Handle snake_case
        name = name.replace("_", " ")

        # Handle camelCase
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return words.lower()

    def _extract_context(self, example: CodeExample) -> str:
        """Extract context from the example."""
        context_parts = []

        if example.description:
            context_parts.append(f"Description: {example.description}")

        if example.context:
            context_parts.append(f"Documentation:\n{example.context}")

        if example.packages:
            packages_str = ", ".join(example.packages)
            context_parts.append(f"Packages: {packages_str}")

        return "\n\n".join(context_parts)

    def _hash_code(self, code: str) -> str:
        """Generate a hash of the code for deduplication."""
        # Normalize whitespace for hashing
        normalized = re.sub(r'\s+', ' ', code.strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _generate_id(self, example: CodeExample) -> str:
        """Generate a unique ID for the example."""
        content = f"{example.source}:{example.source_url}:{example.code[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


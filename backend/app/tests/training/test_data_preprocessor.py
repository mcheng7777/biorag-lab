"""Tests for the DataPreprocessor class."""

import pytest

from app.services.training.data_collector import CodeExample
from app.services.training.data_preprocessor import DataPreprocessor, TrainingExample


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create a DataPreprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_example(self):
        """Create a sample CodeExample."""
        return CodeExample(
            code='''from Bio import SeqIO

def calculate_gc(fasta_file):
    """Calculate GC content for each sequence."""
    results = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq).upper()
        gc = (seq.count("G") + seq.count("C")) / len(seq)
        results[record.id] = gc
    return results
''',
            language="python",
            source="github/biopython/examples",
            source_url="https://github.com/biopython/examples",
            description="GC content calculator",
            packages=["biopython"],
            tags=["bioinformatics", "sequence-analysis"],
        )

    # Preprocessing Tests
    def test_preprocess_valid_example(self, preprocessor, sample_example):
        """Test preprocessing a valid example."""
        result = preprocessor.preprocess(sample_example)

        assert result is not None
        assert isinstance(result, TrainingExample)
        assert result.language == "python"
        assert result.source == sample_example.source
        assert len(result.instruction) > 0
        assert len(result.output) > 0

    def test_preprocess_generates_instruction(self, preprocessor, sample_example):
        """Test that preprocessing generates an instruction."""
        result = preprocessor.preprocess(sample_example)

        assert result.instruction is not None
        # Should contain some keywords from the code
        assert any(word in result.instruction.lower() for word in ["python", "biopython", "calculate", "gc"])

    def test_preprocess_cleans_code(self, preprocessor):
        """Test that preprocessing cleans code."""
        example = CodeExample(
            code='''#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def process():
    return pd.DataFrame()


if __name__ == "__main__":
    process()
''',
            language="python",
            source="test",
            source_url="http://test.com",
        )

        result = preprocessor.preprocess(example)

        # Should remove shebang and encoding
        assert "#!/usr/bin/env" not in result.output
        assert "coding:" not in result.output

    def test_preprocess_deduplication(self, preprocessor):
        """Test that duplicate examples are filtered."""
        example = CodeExample(
            code="import pandas as pd\n\ndef test():\n    return pd.DataFrame()",
            language="python",
            source="test",
            source_url="http://test.com",
        )

        # Process same example twice
        result1 = preprocessor.preprocess(example)
        result2 = preprocessor.preprocess(example)

        assert result1 is not None
        assert result2 is None  # Should be filtered as duplicate

    def test_preprocess_invalid_code(self, preprocessor):
        """Test preprocessing invalid/empty code."""
        example = CodeExample(
            code="",
            language="python",
            source="test",
            source_url="http://test.com",
        )

        result = preprocessor.preprocess(example)
        assert result is None

    # Batch Processing Tests
    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing."""
        examples = [
            CodeExample(
                code="import numpy as np\n\ndef add(a, b):\n    return np.add(a, b)",
                language="python",
                source="test1",
                source_url="http://test1.com",
            ),
            CodeExample(
                code="library(ggplot2)\n\nplot_data <- function(df) {\n    ggplot(df)\n}",
                language="r",
                source="test2",
                source_url="http://test2.com",
            ),
        ]

        results = preprocessor.preprocess_batch(examples)

        assert len(results) == 2
        assert all(isinstance(r, TrainingExample) for r in results)

    # Split Dataset Tests
    def test_split_dataset(self, preprocessor):
        """Test train/val/test splitting."""
        examples = [
            TrainingExample(
                instruction=f"Test instruction {i}",
                input="",
                output=f"Test output {i}",
                language="python",
                source="test",
                example_id=f"example_{i}",
                metadata={},
            )
            for i in range(100)
        ]

        train, val, test = preprocessor.split_dataset(
            examples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
        assert len(train) + len(val) + len(test) == len(examples)

    def test_split_dataset_reproducible(self, preprocessor):
        """Test that splitting is reproducible with same seed."""
        examples = [
            TrainingExample(
                instruction=f"Test {i}",
                input="",
                output=f"Output {i}",
                language="python",
                source="test",
                example_id=f"ex_{i}",
                metadata={},
            )
            for i in range(50)
        ]

        train1, val1, test1 = preprocessor.split_dataset(examples, seed=42)
        train2, val2, test2 = preprocessor.split_dataset(examples, seed=42)

        assert [e.example_id for e in train1] == [e.example_id for e in train2]

    # Chat Format Tests
    def test_to_chat_format(self, preprocessor, sample_example):
        """Test conversion to chat format."""
        result = preprocessor.preprocess(sample_example)
        messages = result.to_chat_format()

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "python" in messages[0]["content"].lower()


class TestInstructionGeneration:
    """Tests for instruction generation."""

    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()

    def test_instruction_from_docstring(self, preprocessor):
        """Test instruction extraction from docstring."""
        example = CodeExample(
            code='''def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers)
''',
            language="python",
            source="test",
            source_url="http://test.com",
        )

        result = preprocessor.preprocess(example)
        # Instruction should be based on docstring content
        assert result is not None
        assert len(result.instruction) > 10

    def test_instruction_from_function_name(self, preprocessor):
        """Test instruction generation from function name."""
        example = CodeExample(
            code='''def process_sequence_data(seq):
    return seq.upper()
''',
            language="python",
            source="test",
            source_url="http://test.com",
            packages=["biopython"],
        )

        result = preprocessor.preprocess(example)
        assert result is not None
        # Should extract task from function name
        assert len(result.instruction) > 0

    def test_r_instruction_generation(self, preprocessor):
        """Test instruction generation for R code."""
        example = CodeExample(
            code='''# Perform differential expression analysis
run_deseq2 <- function(counts, coldata) {
    dds <- DESeqDataSetFromMatrix(counts, coldata, ~ condition)
    dds <- DESeq(dds)
    return(results(dds))
}
''',
            language="r",
            source="test",
            source_url="http://test.com",
            packages=["DESeq2"],
        )

        result = preprocessor.preprocess(example)
        assert result is not None
        assert "r" in result.instruction.lower() or "deseq2" in result.instruction.lower()


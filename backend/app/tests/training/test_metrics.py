"""Tests for code metrics computation."""

import pytest

from app.services.training.evaluation.metrics import (
    CodeMetrics,
    compute_code_metrics,
    aggregate_metrics,
)


class TestCodeMetrics:
    """Tests for CodeMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = CodeMetrics()

        assert metrics.syntax_valid is False
        assert metrics.execution_success is False
        assert metrics.has_imports is False
        assert metrics.overall_score == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = CodeMetrics(
            syntax_valid=True,
            has_imports=True,
            num_lines=50,
        )

        d = metrics.to_dict()

        assert d["syntax_valid"] is True
        assert d["has_imports"] is True
        assert d["num_lines"] == 50


class TestComputeCodeMetrics:
    """Tests for compute_code_metrics function."""

    # Python Metrics Tests
    def test_python_valid_code(self):
        """Test metrics for valid Python code."""
        code = """
import pandas as pd

def process_data(df):
    \"\"\"Process the dataframe.\"\"\"
    return df.dropna()
"""
        metrics = compute_code_metrics(code, "python")

        assert metrics.syntax_valid is True
        assert metrics.has_imports is True
        assert metrics.has_functions is True
        assert metrics.num_functions == 1
        assert "pandas" in metrics.detected_packages

    def test_python_with_docstring(self):
        """Test that docstrings are detected."""
        code = '''
def calculate(x):
    """
    Calculate something.
    
    Args:
        x: Input value
    
    Returns:
        Calculated result
    """
    return x * 2
'''
        metrics = compute_code_metrics(code, "python")

        assert metrics.syntax_valid is True
        assert metrics.has_docstrings is True

    def test_python_syntax_error(self):
        """Test metrics for code with syntax error."""
        code = """
def broken(
    return x
"""
        metrics = compute_code_metrics(code, "python")

        assert metrics.syntax_valid is False
        assert metrics.syntax_error is not None

    def test_python_multiple_imports(self):
        """Test detection of multiple imports."""
        code = """
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
"""
        metrics = compute_code_metrics(code, "python")

        assert metrics.has_imports is True
        assert "numpy" in metrics.detected_packages
        assert "pandas" in metrics.detected_packages
        assert "Bio" in metrics.detected_packages

    def test_python_class_detection(self):
        """Test class detection."""
        code = """
class DataProcessor:
    def __init__(self):
        pass
    
    def process(self, data):
        return data
"""
        metrics = compute_code_metrics(code, "python")

        assert metrics.num_classes == 1
        assert metrics.num_functions == 2  # __init__ and process

    # R Metrics Tests
    def test_r_valid_code(self):
        """Test metrics for valid R code."""
        code = """
library(DESeq2)

run_analysis <- function(counts) {
    dds <- DESeqDataSetFromMatrix(counts)
    return(dds)
}
"""
        metrics = compute_code_metrics(code, "r")

        assert metrics.syntax_valid is True
        assert metrics.has_imports is True
        assert metrics.has_functions is True
        assert "DESeq2" in metrics.detected_packages

    def test_r_multiple_libraries(self):
        """Test R code with multiple libraries."""
        code = """
library(DESeq2)
library(ggplot2)
require(dplyr)
"""
        metrics = compute_code_metrics(code, "r")

        assert metrics.has_imports is True
        assert "DESeq2" in metrics.detected_packages
        assert "ggplot2" in metrics.detected_packages
        assert "dplyr" in metrics.detected_packages

    def test_r_unbalanced_braces(self):
        """Test R code with syntax error."""
        code = """
test <- function(x) {
    return(x
}
"""
        metrics = compute_code_metrics(code, "r")

        assert metrics.syntax_valid is False

    # Expected Packages Tests
    def test_expected_packages_found(self):
        """Test that expected packages are detected."""
        code = """
from Bio import SeqIO
import pandas as pd
"""
        metrics = compute_code_metrics(
            code, "python",
            expected_packages=["biopython", "pandas"]
        )

        assert metrics.expected_packages_used is True

    def test_expected_packages_missing(self):
        """Test when expected packages are not found."""
        code = """
import numpy as np
"""
        metrics = compute_code_metrics(
            code, "python",
            expected_packages=["pandas", "biopython"]
        )

        assert metrics.expected_packages_used is False

    # Quality Score Tests
    def test_high_quality_score(self):
        """Test that good code gets high quality score."""
        code = '''
import pandas as pd
from Bio import SeqIO

def analyze_sequences(fasta_file):
    """
    Analyze sequences from a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        
    Returns:
        DataFrame with sequence statistics
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    data = []
    for record in records:
        data.append({
            "id": record.id,
            "length": len(record.seq),
        })
    
    return pd.DataFrame(data)
'''
        metrics = compute_code_metrics(code, "python")

        assert metrics.overall_score >= 0.5

    def test_low_quality_score(self):
        """Test that minimal code gets low quality score."""
        code = "x = 1"

        metrics = compute_code_metrics(code, "python")

        # Minimal code should have low score (syntax valid adds 0.3)
        assert metrics.overall_score <= 0.4

    # Edge Cases
    def test_empty_code(self):
        """Test metrics for empty code."""
        metrics = compute_code_metrics("", "python")

        assert metrics.syntax_valid is False
        assert metrics.overall_score == 0.0

    def test_whitespace_only(self):
        """Test metrics for whitespace-only code."""
        metrics = compute_code_metrics("   \n\n   ", "python")

        assert metrics.num_lines == 0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_aggregate_single(self):
        """Test aggregation of single metric."""
        metrics = [
            CodeMetrics(syntax_valid=True, overall_score=0.8),
        ]

        result = aggregate_metrics(metrics)

        assert result["count"] == 1
        assert result["syntax_valid_rate"] == 1.0
        assert result["avg_overall_score"] == 0.8

    def test_aggregate_multiple(self):
        """Test aggregation of multiple metrics."""
        metrics = [
            CodeMetrics(syntax_valid=True, has_imports=True, overall_score=0.8),
            CodeMetrics(syntax_valid=True, has_imports=False, overall_score=0.6),
            CodeMetrics(syntax_valid=False, has_imports=False, overall_score=0.2),
        ]

        result = aggregate_metrics(metrics)

        assert result["count"] == 3
        assert result["syntax_valid_rate"] == pytest.approx(2/3)
        assert result["has_imports_rate"] == pytest.approx(1/3)
        assert result["avg_overall_score"] == pytest.approx((0.8 + 0.6 + 0.2) / 3)

    def test_aggregate_empty(self):
        """Test aggregation of empty list."""
        result = aggregate_metrics([])

        assert result == {}


"""
Code quality metrics for evaluating generated bioinformatics code.

Metrics include:
- Syntax correctness
- Code execution success
- Package usage accuracy
- Documentation quality
"""

import ast
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Metrics for evaluating code quality."""

    # Syntax metrics
    syntax_valid: bool = False
    syntax_error: Optional[str] = None

    # Execution metrics
    execution_success: bool = False
    execution_error: Optional[str] = None
    execution_output: Optional[str] = None

    # Structure metrics
    has_imports: bool = False
    has_functions: bool = False
    has_docstrings: bool = False
    num_lines: int = 0
    num_functions: int = 0
    num_classes: int = 0

    # Package metrics
    detected_packages: list[str] = field(default_factory=list)
    expected_packages_used: bool = True

    # Documentation metrics
    comment_ratio: float = 0.0
    has_type_hints: bool = False

    # Overall score (0-1)
    overall_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "syntax_valid": self.syntax_valid,
            "syntax_error": self.syntax_error,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "has_imports": self.has_imports,
            "has_functions": self.has_functions,
            "has_docstrings": self.has_docstrings,
            "num_lines": self.num_lines,
            "num_functions": self.num_functions,
            "num_classes": self.num_classes,
            "detected_packages": self.detected_packages,
            "comment_ratio": self.comment_ratio,
            "has_type_hints": self.has_type_hints,
            "overall_score": self.overall_score,
        }


def compute_code_metrics(
    code: str,
    language: str,
    expected_packages: Optional[list[str]] = None,
    execute_code: bool = False,
    timeout: int = 30,
) -> CodeMetrics:
    """
    Compute comprehensive metrics for a code sample.

    Args:
        code: The code to evaluate
        language: "python" or "r"
        expected_packages: List of packages expected to be used
        execute_code: Whether to attempt code execution
        timeout: Timeout for code execution in seconds

    Returns:
        CodeMetrics with evaluation results
    """
    metrics = CodeMetrics()

    if not code or not code.strip():
        return metrics

    # Basic structure metrics
    lines = code.strip().split("\n")
    metrics.num_lines = len(lines)

    # Language-specific metrics
    if language == "python":
        _compute_python_metrics(code, metrics)
    else:
        _compute_r_metrics(code, metrics)

    # Package detection
    metrics.detected_packages = _detect_packages(code, language)

    # Check expected packages
    if expected_packages:
        expected_set = set(p.lower() for p in expected_packages)
        detected_set = set(p.lower() for p in metrics.detected_packages)
        metrics.expected_packages_used = bool(expected_set & detected_set)

    # Execute code if requested
    if execute_code and metrics.syntax_valid:
        _execute_code(code, language, metrics, timeout)

    # Calculate overall score
    metrics.overall_score = _calculate_score(metrics)

    return metrics


def _compute_python_metrics(code: str, metrics: CodeMetrics) -> None:
    """Compute Python-specific metrics."""
    # Syntax check
    try:
        tree = ast.parse(code)
        metrics.syntax_valid = True

        # Count functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics.num_functions += 1
                # Check for docstring
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    metrics.has_docstrings = True
            elif isinstance(node, ast.ClassDef):
                metrics.num_classes += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics.has_imports = True

        metrics.has_functions = metrics.num_functions > 0

        # Check for type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    metrics.has_type_hints = True
                    break

    except SyntaxError as e:
        metrics.syntax_valid = False
        metrics.syntax_error = str(e)

    # Comment ratio
    lines = code.split("\n")
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    metrics.comment_ratio = comment_lines / max(len(lines), 1)


def _compute_r_metrics(code: str, metrics: CodeMetrics) -> None:
    """Compute R-specific metrics."""
    # Basic syntax check (regex-based)
    metrics.syntax_valid = _check_r_syntax(code)

    # Check for library calls
    if re.search(r"library\s*\(|require\s*\(", code):
        metrics.has_imports = True

    # Check for function definitions
    func_matches = re.findall(r"(\w+)\s*<-\s*function\s*\(", code)
    metrics.num_functions = len(func_matches)
    metrics.has_functions = metrics.num_functions > 0

    # Check for documentation (roxygen2 style)
    if re.search(r"#'", code):
        metrics.has_docstrings = True

    # Comment ratio
    lines = code.split("\n")
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    metrics.comment_ratio = comment_lines / max(len(lines), 1)


def _check_r_syntax(code: str) -> bool:
    """Basic R syntax validation."""
    # Check for balanced brackets
    if code.count("(") != code.count(")"):
        return False
    if code.count("{") != code.count("}"):
        return False
    if code.count("[") != code.count("]"):
        return False

    return True


def _detect_packages(code: str, language: str) -> list[str]:
    """Detect packages used in code."""
    packages = set()

    if language == "python":
        # import X
        for match in re.finditer(r"^import\s+([\w.]+)", code, re.MULTILINE):
            packages.add(match.group(1).split(".")[0])

        # from X import
        for match in re.finditer(r"^from\s+([\w.]+)\s+import", code, re.MULTILINE):
            packages.add(match.group(1).split(".")[0])

    else:  # R
        # library(X) or require(X)
        for match in re.finditer(
            r"(?:library|require)\s*\(\s*['\"]?(\w+)['\"]?\s*\)", code
        ):
            packages.add(match.group(1))

    return list(packages)


def _execute_code(
    code: str,
    language: str,
    metrics: CodeMetrics,
    timeout: int,
) -> None:
    """Attempt to execute code in a sandboxed environment."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            if language == "python":
                file_path = Path(tmpdir) / "test_code.py"
                file_path.write_text(code)

                result = subprocess.run(
                    ["python", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )

            else:  # R
                file_path = Path(tmpdir) / "test_code.R"
                file_path.write_text(code)

                result = subprocess.run(
                    ["Rscript", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )

            if result.returncode == 0:
                metrics.execution_success = True
                metrics.execution_output = result.stdout[:1000]  # Limit output size
            else:
                metrics.execution_success = False
                metrics.execution_error = result.stderr[:500]

    except subprocess.TimeoutExpired:
        metrics.execution_success = False
        metrics.execution_error = f"Execution timed out after {timeout}s"

    except FileNotFoundError as e:
        metrics.execution_success = False
        metrics.execution_error = f"Interpreter not found: {e}"

    except Exception as e:
        metrics.execution_success = False
        metrics.execution_error = str(e)


def _calculate_score(metrics: CodeMetrics) -> float:
    """
    Calculate overall code quality score.

    Scoring:
    - Syntax valid: 30%
    - Has imports: 10%
    - Has functions: 15%
    - Has docstrings: 10%
    - Comment ratio (10-30%): 10%
    - Execution success: 25%
    """
    score = 0.0

    # Syntax (30%)
    if metrics.syntax_valid:
        score += 0.3

    # Imports (10%)
    if metrics.has_imports:
        score += 0.1

    # Functions (15%)
    if metrics.has_functions:
        score += 0.15

    # Docstrings (10%)
    if metrics.has_docstrings:
        score += 0.1

    # Comment ratio (10%)
    if 0.1 <= metrics.comment_ratio <= 0.3:
        score += 0.1
    elif 0.05 <= metrics.comment_ratio <= 0.4:
        score += 0.05

    # Execution (25%)
    if metrics.execution_success:
        score += 0.25

    return round(score, 2)


def aggregate_metrics(metrics_list: list[CodeMetrics]) -> dict:
    """
    Aggregate metrics across multiple code samples.

    Args:
        metrics_list: List of CodeMetrics

    Returns:
        Dictionary with aggregated statistics
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    return {
        "count": n,
        "syntax_valid_rate": sum(1 for m in metrics_list if m.syntax_valid) / n,
        "execution_success_rate": sum(1 for m in metrics_list if m.execution_success) / n,
        "has_imports_rate": sum(1 for m in metrics_list if m.has_imports) / n,
        "has_functions_rate": sum(1 for m in metrics_list if m.has_functions) / n,
        "has_docstrings_rate": sum(1 for m in metrics_list if m.has_docstrings) / n,
        "avg_num_lines": sum(m.num_lines for m in metrics_list) / n,
        "avg_num_functions": sum(m.num_functions for m in metrics_list) / n,
        "avg_comment_ratio": sum(m.comment_ratio for m in metrics_list) / n,
        "avg_overall_score": sum(m.overall_score for m in metrics_list) / n,
    }


"""
Data validation for collected bioinformatics code examples.

Validates:
- Code syntax correctness
- Package imports
- Code quality metrics
- Bioinformatics-specific patterns
"""

import ast
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of code validation."""

    status: ValidationStatus
    syntax_valid: bool
    has_imports: bool
    has_functions: bool
    code_quality_score: float  # 0.0 to 1.0
    issues: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID


class DataValidator:
    """
    Validates code examples for training data quality.
    """

    # Minimum requirements for valid code
    MIN_CODE_LENGTH = 50
    MIN_CODE_LINES = 3
    MAX_CODE_LENGTH = 50000
    MIN_QUALITY_SCORE = 0.3

    # Bioinformatics-specific packages
    PYTHON_BIO_PACKAGES = {
        "biopython": ["Bio", "Bio.Seq", "Bio.SeqIO", "Bio.Align", "Bio.Blast"],
        "scanpy": ["scanpy", "sc"],
        "anndata": ["anndata", "ad"],
        "pandas": ["pandas", "pd"],
        "numpy": ["numpy", "np"],
        "scikit-bio": ["skbio"],
        "pysam": ["pysam"],
        "pyvcf": ["vcf"],
        "scipy": ["scipy"],
        "matplotlib": ["matplotlib", "plt"],
        "seaborn": ["seaborn", "sns"],
    }

    R_BIO_PACKAGES = {
        "deseq2": ["DESeq2"],
        "edger": ["edgeR"],
        "limma": ["limma"],
        "complexheatmap": ["ComplexHeatmap"],
        "seurat": ["Seurat"],
        "ggplot2": ["ggplot2"],
        "dplyr": ["dplyr"],
        "bioconductor": ["BiocManager", "Biobase"],
        "genomicranges": ["GenomicRanges", "IRanges"],
        "biostrings": ["Biostrings"],
    }

    def validate(self, code: str, language: str) -> ValidationResult:
        """
        Validate a code example.

        Args:
            code: The code to validate
            language: "python" or "r"

        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []

        # Basic length checks
        if len(code) < self.MIN_CODE_LENGTH:
            issues.append(f"Code too short ({len(code)} chars, min {self.MIN_CODE_LENGTH})")

        if len(code) > self.MAX_CODE_LENGTH:
            issues.append(f"Code too long ({len(code)} chars, max {self.MAX_CODE_LENGTH})")

        lines = code.strip().split("\n")
        if len(lines) < self.MIN_CODE_LINES:
            issues.append(f"Too few lines ({len(lines)}, min {self.MIN_CODE_LINES})")

        # Syntax validation
        syntax_valid = self._check_syntax(code, language)
        if not syntax_valid:
            issues.append("Syntax errors detected")

        # Import/library checks
        has_imports = self._has_imports(code, language)
        if not has_imports:
            warnings.append("No imports/libraries detected")

        # Function/class definitions
        has_functions = self._has_functions(code, language)

        # Bioinformatics package check
        bio_packages = self._detect_bio_packages(code, language)
        if not bio_packages:
            warnings.append("No bioinformatics packages detected")

        # Code quality score
        quality_score = self._calculate_quality_score(
            code, language, syntax_valid, has_imports, has_functions, bio_packages
        )

        if quality_score < self.MIN_QUALITY_SCORE:
            issues.append(f"Quality score too low ({quality_score:.2f}, min {self.MIN_QUALITY_SCORE})")

        # Determine overall status
        if issues:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return ValidationResult(
            status=status,
            syntax_valid=syntax_valid,
            has_imports=has_imports,
            has_functions=has_functions,
            code_quality_score=quality_score,
            issues=issues,
            warnings=warnings,
        )

    def _check_syntax(self, code: str, language: str) -> bool:
        """Check if code has valid syntax."""
        if language == "python":
            return self._check_python_syntax(code)
        elif language == "r":
            return self._check_r_syntax(code)
        return False

    def _check_python_syntax(self, code: str) -> bool:
        """Check Python syntax using AST."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _check_r_syntax(self, code: str) -> bool:
        """
        Basic R syntax check using pattern matching.
        Note: Full R syntax checking would require an R interpreter.
        """
        # Check for common syntax errors
        error_patterns = [
            r'\)\s*\{[^\}]*$',  # Unclosed braces
            r'\[\s*$',  # Unclosed brackets
            r'^\s*\)',  # Orphaned closing parenthesis
        ]

        for pattern in error_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return False

        # Check for balanced parentheses
        if code.count("(") != code.count(")"):
            return False

        # Check for balanced braces
        if code.count("{") != code.count("}"):
            return False

        # Check for balanced brackets
        if code.count("[") != code.count("]"):
            return False

        return True

    def _has_imports(self, code: str, language: str) -> bool:
        """Check if code has import statements."""
        if language == "python":
            patterns = [r"^\s*import\s+", r"^\s*from\s+\w+\s+import"]
        else:  # R
            patterns = [r"library\s*\(", r"require\s*\(", r"source\s*\("]

        return any(re.search(p, code, re.MULTILINE) for p in patterns)

    def _has_functions(self, code: str, language: str) -> bool:
        """Check if code defines functions or classes."""
        if language == "python":
            patterns = [r"^\s*def\s+\w+", r"^\s*class\s+\w+"]
        else:  # R
            patterns = [r"\w+\s*<-\s*function\s*\(", r"function\s*\("]

        return any(re.search(p, code, re.MULTILINE) for p in patterns)

    def _detect_bio_packages(self, code: str, language: str) -> list[str]:
        """Detect bioinformatics packages in code."""
        detected = []
        packages = self.PYTHON_BIO_PACKAGES if language == "python" else self.R_BIO_PACKAGES

        for package_name, aliases in packages.items():
            for alias in aliases:
                if language == "python":
                    pattern = rf"(?:import|from)\s+{re.escape(alias)}"
                else:
                    pattern = rf"library\s*\(\s*['\"]?{re.escape(alias)}['\"]?\s*\)"

                if re.search(pattern, code):
                    detected.append(package_name)
                    break

        return detected

    def _calculate_quality_score(
        self,
        code: str,
        language: str,
        syntax_valid: bool,
        has_imports: bool,
        has_functions: bool,
        bio_packages: list[str],
    ) -> float:
        """
        Calculate a code quality score from 0.0 to 1.0.

        Scoring factors:
        - Syntax validity (0.3)
        - Has imports (0.15)
        - Has functions (0.15)
        - Bioinformatics packages (0.2)
        - Code structure (0.2)
        """
        score = 0.0

        # Syntax validity (30%)
        if syntax_valid:
            score += 0.3

        # Has imports (15%)
        if has_imports:
            score += 0.15

        # Has functions (15%)
        if has_functions:
            score += 0.15

        # Bioinformatics packages (20%)
        if bio_packages:
            # More bio packages = higher score, max at 3 packages
            bio_score = min(len(bio_packages) / 3.0, 1.0) * 0.2
            score += bio_score

        # Code structure (20%)
        structure_score = self._evaluate_structure(code, language)
        score += structure_score * 0.2

        return round(score, 2)

    def _evaluate_structure(self, code: str, language: str) -> float:
        """
        Evaluate code structure quality.

        Checks:
        - Comment ratio
        - Line length distribution
        - Docstrings (Python)
        - Variable naming
        """
        score = 0.0
        lines = code.strip().split("\n")

        # Comment ratio (target: 10-30% comments)
        if language == "python":
            comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        else:
            comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

        comment_ratio = comment_lines / max(len(lines), 1)
        if 0.1 <= comment_ratio <= 0.3:
            score += 0.3
        elif 0.05 <= comment_ratio <= 0.4:
            score += 0.15

        # Line length (prefer lines under 100 chars)
        long_lines = sum(1 for l in lines if len(l) > 100)
        long_line_ratio = long_lines / max(len(lines), 1)
        if long_line_ratio < 0.1:
            score += 0.3
        elif long_line_ratio < 0.2:
            score += 0.15

        # Docstrings (Python) or function documentation (R)
        if language == "python":
            if '"""' in code or "'''" in code:
                score += 0.2
        else:
            if "#'" in code or "roxygen" in code.lower():
                score += 0.2

        # Non-trivial code (has loops, conditionals)
        if language == "python":
            control_patterns = [r"\bfor\b", r"\bwhile\b", r"\bif\b", r"\btry\b"]
        else:
            control_patterns = [r"\bfor\b", r"\bwhile\b", r"\bif\b", r"\btryCatch\b"]

        control_count = sum(
            1 for p in control_patterns if re.search(p, code)
        )
        if control_count >= 2:
            score += 0.2
        elif control_count >= 1:
            score += 0.1

        return min(score, 1.0)

    def validate_batch(
        self,
        examples: list,  # list[CodeExample]
        min_quality: Optional[float] = None,
    ) -> tuple[list, list, dict]:
        """
        Validate a batch of examples.

        Args:
            examples: List of CodeExample objects
            min_quality: Minimum quality score to keep (default: MIN_QUALITY_SCORE)

        Returns:
            Tuple of (valid_examples, invalid_examples, stats)
        """
        if min_quality is None:
            min_quality = self.MIN_QUALITY_SCORE

        valid = []
        invalid = []
        stats = {
            "total": len(examples),
            "valid": 0,
            "invalid": 0,
            "warnings": 0,
            "avg_quality": 0.0,
            "by_language": {"python": 0, "r": 0},
            "by_issue": {},
        }

        total_quality = 0.0

        for example in examples:
            result = self.validate(example.code, example.language)
            total_quality += result.code_quality_score

            if result.is_valid and result.code_quality_score >= min_quality:
                valid.append(example)
                stats["valid"] += 1
                stats["by_language"][example.language] += 1
            else:
                invalid.append(example)
                stats["invalid"] += 1
                for issue in result.issues:
                    stats["by_issue"][issue] = stats["by_issue"].get(issue, 0) + 1

            if result.status == ValidationStatus.WARNING:
                stats["warnings"] += 1

        stats["avg_quality"] = total_quality / max(len(examples), 1)

        logger.info(
            f"Validation complete: {stats['valid']}/{stats['total']} valid "
            f"(avg quality: {stats['avg_quality']:.2f})"
        )

        return valid, invalid, stats


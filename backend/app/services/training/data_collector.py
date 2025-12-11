"""
Data collection services for gathering bioinformatics code from various sources.

Supports collecting code from:
- GitHub repositories (Bioconductor, Biopython, etc.)
- Package documentation examples
- Paper implementation repositories
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CodeExample:
    """Represents a collected code example."""

    code: str
    language: str  # "python" or "r"
    source: str  # e.g., "github/biopython/biopython"
    source_url: str
    description: Optional[str] = None
    context: Optional[str] = None  # Documentation or paper context
    packages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    collected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "language": self.language,
            "source": self.source,
            "source_url": self.source_url,
            "description": self.description,
            "context": self.context,
            "packages": self.packages,
            "tags": self.tags,
            "collected_at": self.collected_at.isoformat(),
        }


class BaseCollector(ABC):
    """Abstract base class for code collectors."""

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Collector must be used as async context manager")
        return self._client

    @abstractmethod
    async def collect(self, max_examples: int = 100) -> list[CodeExample]:
        """Collect code examples from the source."""
        pass

    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        await asyncio.sleep(self.rate_limit_delay)


class GitHubCollector(BaseCollector):
    """Collect code from GitHub repositories."""

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limit_delay: float = 1.0,
    ):
        super().__init__(rate_limit_delay)
        self.token = token
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    async def search_code(
        self,
        query: str,
        language: str = "python",
        max_results: int = 100,
    ) -> list[dict]:
        """Search GitHub for code matching the query."""
        results = []
        page = 1
        per_page = min(100, max_results)

        while len(results) < max_results:
            search_query = f"{query} language:{language}"
            url = f"{self.GITHUB_API_BASE}/search/code"
            params = {
                "q": search_query,
                "per_page": per_page,
                "page": page,
            }

            try:
                response = await self.client.get(
                    url, headers=self.headers, params=params
                )
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    break

                results.extend(items)
                page += 1
                await self._rate_limit()

            except httpx.HTTPStatusError as e:
                logger.error(f"GitHub API error: {e}")
                break
            except Exception as e:
                logger.error(f"Error searching GitHub: {e}")
                break

        return results[:max_results]

    async def get_file_content(self, url: str) -> Optional[str]:
        """Get the raw content of a file from GitHub."""
        try:
            # Convert API URL to raw URL
            raw_url = url.replace(
                "https://api.github.com/repos", "https://raw.githubusercontent.com"
            ).replace("/contents/", "/main/")

            response = await self.client.get(raw_url, headers=self.headers)
            response.raise_for_status()
            return response.text

        except Exception as e:
            logger.error(f"Error fetching file content: {e}")
            return None

    async def collect(self, max_examples: int = 100) -> list[CodeExample]:
        """Collect bioinformatics code examples from GitHub."""
        examples = []

        # Search queries for bioinformatics code
        queries = [
            "biopython sequence analysis",
            "bioconductor DESeq2",
            "bioconductor ComplexHeatmap",
            "scanpy single cell",
            "seurat RNA-seq",
            "pandas genomics",
            "scikit-bio",
            "pysam BAM",
            "pyvcf VCF",
            "anndata single cell",
        ]

        examples_per_query = max(1, max_examples // len(queries))

        for query in queries:
            if len(examples) >= max_examples:
                break

            # Search for Python code
            python_results = await self.search_code(
                query, language="python", max_results=examples_per_query
            )

            for item in python_results:
                if len(examples) >= max_examples:
                    break

                content = await self.get_file_content(item.get("url", ""))
                if content and self._is_valid_code(content, "python"):
                    packages = self._extract_packages(content, "python")
                    example = CodeExample(
                        code=content,
                        language="python",
                        source=f"github/{item.get('repository', {}).get('full_name', 'unknown')}",
                        source_url=item.get("html_url", ""),
                        description=item.get("name", ""),
                        packages=packages,
                        tags=["bioinformatics", "github"],
                    )
                    examples.append(example)
                    await self._rate_limit()

            # Search for R code
            r_results = await self.search_code(
                query, language="r", max_results=examples_per_query
            )

            for item in r_results:
                if len(examples) >= max_examples:
                    break

                content = await self.get_file_content(item.get("url", ""))
                if content and self._is_valid_code(content, "r"):
                    packages = self._extract_packages(content, "r")
                    example = CodeExample(
                        code=content,
                        language="r",
                        source=f"github/{item.get('repository', {}).get('full_name', 'unknown')}",
                        source_url=item.get("html_url", ""),
                        description=item.get("name", ""),
                        packages=packages,
                        tags=["bioinformatics", "github"],
                    )
                    examples.append(example)
                    await self._rate_limit()

        logger.info(f"Collected {len(examples)} examples from GitHub")
        return examples

    def _is_valid_code(self, content: str, language: str) -> bool:
        """Check if the content is valid code."""
        if not content or len(content) < 50:
            return False

        # Check for minimum code indicators
        if language == "python":
            indicators = ["import", "def ", "class ", "from "]
        else:  # R
            indicators = ["library(", "function(", "<-", "source("]

        return any(indicator in content for indicator in indicators)

    def _extract_packages(self, content: str, language: str) -> list[str]:
        """Extract package names from code."""
        packages = set()

        if language == "python":
            # Match import statements
            import_pattern = r"(?:from|import)\s+([\w.]+)"
            matches = re.findall(import_pattern, content)
            for match in matches:
                # Get the top-level package
                packages.add(match.split(".")[0])
        else:  # R
            # Match library() and require() calls
            lib_pattern = r"(?:library|require)\s*\(\s*['\"]?(\w+)['\"]?\s*\)"
            matches = re.findall(lib_pattern, content)
            packages.update(matches)

        return list(packages)


class DocumentationCollector(BaseCollector):
    """Collect code examples from package documentation."""

    def __init__(self, rate_limit_delay: float = 1.0):
        super().__init__(rate_limit_delay)

    async def collect_from_url(
        self,
        url: str,
        language: str,
        package_name: str,
    ) -> list[CodeExample]:
        """Collect code examples from a documentation URL."""
        examples = []

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            content = response.text

            # Extract code blocks
            code_blocks = self._extract_code_blocks(content, language)

            for i, code in enumerate(code_blocks):
                if self._is_valid_example(code):
                    example = CodeExample(
                        code=code,
                        language=language,
                        source=f"docs/{package_name}",
                        source_url=url,
                        description=f"Documentation example {i + 1}",
                        packages=[package_name],
                        tags=["documentation", "bioinformatics"],
                    )
                    examples.append(example)

        except Exception as e:
            logger.error(f"Error collecting from {url}: {e}")

        return examples

    def _extract_code_blocks(self, content: str, language: str) -> list[str]:
        """Extract code blocks from HTML/Markdown content."""
        code_blocks = []

        # Match fenced code blocks (Markdown)
        if language == "python":
            patterns = [
                r"```python\n(.*?)```",
                r"```py\n(.*?)```",
                r"<pre><code class=\"python\">(.*?)</code></pre>",
            ]
        else:
            patterns = [
                r"```r\n(.*?)```",
                r"```R\n(.*?)```",
                r"<pre><code class=\"r\">(.*?)</code></pre>",
            ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            code_blocks.extend(matches)

        return code_blocks

    def _is_valid_example(self, code: str) -> bool:
        """Check if code is a valid, substantial example."""
        # Filter out very short examples
        if len(code.strip()) < 50:
            return False

        # Filter out examples that are mostly comments
        lines = code.strip().split("\n")
        code_lines = [
            line
            for line in lines
            if line.strip() and not line.strip().startswith(("#", "//"))
        ]
        return len(code_lines) >= 3

    async def collect(self, max_examples: int = 100) -> list[CodeExample]:
        """Collect examples from known documentation sources."""
        examples = []

        # This is a placeholder - in production, you would have a list of
        # documentation URLs to scrape
        logger.info("Documentation collector requires specific URLs")
        return examples


class BioconductorCollector(DocumentationCollector):
    """Collect code examples from Bioconductor package vignettes."""

    BIOCONDUCTOR_BASE = "https://bioconductor.org"

    # Popular Bioconductor packages for bioinformatics
    PACKAGES = [
        "DESeq2",
        "edgeR",
        "limma",
        "ComplexHeatmap",
        "clusterProfiler",
        "GenomicRanges",
        "Biostrings",
        "AnnotationDbi",
        "SingleCellExperiment",
        "scater",
        "scran",
        "Seurat",
    ]

    async def get_vignette_urls(self, package: str) -> list[str]:
        """Get vignette URLs for a Bioconductor package."""
        urls = []
        package_url = f"{self.BIOCONDUCTOR_BASE}/packages/release/bioc/html/{package}.html"

        try:
            response = await self.client.get(package_url)
            response.raise_for_status()
            content = response.text

            # Extract vignette links
            vignette_pattern = rf'/packages/release/bioc/vignettes/{package}/inst/doc/[\w.-]+\.html'
            matches = re.findall(vignette_pattern, content)
            urls = [f"{self.BIOCONDUCTOR_BASE}{m}" for m in matches]

        except Exception as e:
            logger.error(f"Error getting vignettes for {package}: {e}")

        return urls

    async def collect(self, max_examples: int = 100) -> list[CodeExample]:
        """Collect R code examples from Bioconductor vignettes."""
        examples = []
        examples_per_package = max(1, max_examples // len(self.PACKAGES))

        for package in self.PACKAGES:
            if len(examples) >= max_examples:
                break

            vignette_urls = await self.get_vignette_urls(package)
            await self._rate_limit()

            for url in vignette_urls[:2]:  # Limit to 2 vignettes per package
                if len(examples) >= max_examples:
                    break

                package_examples = await self.collect_from_url(url, "r", package)
                examples.extend(package_examples[:examples_per_package])
                await self._rate_limit()

        logger.info(f"Collected {len(examples)} examples from Bioconductor")
        return examples


class BiopythonCollector(DocumentationCollector):
    """Collect code examples from Biopython documentation."""

    BIOPYTHON_DOCS = "https://biopython.org/wiki/Category:Cookbook"

    # Key Biopython modules
    MODULES = [
        "Bio.Seq",
        "Bio.SeqIO",
        "Bio.Align",
        "Bio.Blast",
        "Bio.Entrez",
        "Bio.PDB",
        "Bio.Phylo",
        "Bio.motifs",
    ]

    async def collect(self, max_examples: int = 100) -> list[CodeExample]:
        """Collect Python code examples from Biopython documentation."""
        examples = []

        # Collect from Biopython cookbook
        try:
            response = await self.client.get(self.BIOPYTHON_DOCS)
            response.raise_for_status()
            content = response.text

            # Extract cookbook page links
            cookbook_pattern = r'/wiki/([\w_]+)'
            pages = re.findall(cookbook_pattern, content)

            for page in pages[:max_examples]:
                if len(examples) >= max_examples:
                    break

                page_url = f"https://biopython.org/wiki/{page}"
                page_examples = await self.collect_from_url(
                    page_url, "python", "biopython"
                )
                examples.extend(page_examples)
                await self._rate_limit()

        except Exception as e:
            logger.error(f"Error collecting from Biopython: {e}")

        logger.info(f"Collected {len(examples)} examples from Biopython")
        return examples


class DataCollector:
    """
    Main data collector that orchestrates collection from multiple sources.
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        output_dir: str = "data/training",
    ):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def collect_all(
        self,
        max_examples: int = 1000,
        sources: Optional[list[str]] = None,
    ) -> list[CodeExample]:
        """
        Collect code examples from all configured sources.

        Args:
            max_examples: Maximum total examples to collect
            sources: List of sources to collect from. Options:
                     ["github", "bioconductor", "biopython", "documentation"]
                     If None, collects from all sources.

        Returns:
            List of collected CodeExample objects
        """
        if sources is None:
            sources = ["github", "bioconductor", "biopython"]

        all_examples = []
        examples_per_source = max_examples // len(sources)

        collectors = {
            "github": GitHubCollector(token=self.github_token),
            "bioconductor": BioconductorCollector(),
            "biopython": BiopythonCollector(),
            "documentation": DocumentationCollector(),
        }

        for source in sources:
            if source not in collectors:
                logger.warning(f"Unknown source: {source}")
                continue

            collector = collectors[source]
            async with collector:
                try:
                    examples = await collector.collect(max_examples=examples_per_source)
                    all_examples.extend(examples)
                    logger.info(f"Collected {len(examples)} from {source}")
                except Exception as e:
                    logger.error(f"Error collecting from {source}: {e}")

        logger.info(f"Total collected: {len(all_examples)} examples")
        return all_examples

    def save_examples(
        self,
        examples: list[CodeExample],
        filename: str = "collected_examples.jsonl",
    ) -> Path:
        """Save collected examples to a JSONL file."""
        import json

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return output_path

    def load_examples(self, filename: str = "collected_examples.jsonl") -> list[CodeExample]:
        """Load examples from a JSONL file."""
        import json

        input_path = self.output_dir / filename
        examples = []

        with open(input_path, "r") as f:
            for line in f:
                data = json.loads(line)
                # Convert collected_at back to datetime
                data["collected_at"] = datetime.fromisoformat(data["collected_at"])
                examples.append(CodeExample(**data))

        logger.info(f"Loaded {len(examples)} examples from {input_path}")
        return examples


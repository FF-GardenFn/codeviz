from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import ast
from typing import Dict, Set, List, Optional, Any, Type
import re

import structlog
from . import REGISTRY
from .base import FileAnalyzer
from ..models.file_summary import FileSummary
from ..models.report import Report, ContextSummary
from ..services.directory_tree import DirectoryTree
from ..services.openai_embeddings import embed
from ..services.context_summarizer import ContextSummarizer

logger = structlog.get_logger(__name__)


class ProjectAnalyzer:
    """
    Orchestrates concrete analyzers to build the dependency graph and analysis.
    """

    def __init__(self, root: Path):
        """
        Initialize the project analyzer.

        Args:
            root: Path to the project root directory
        """
        self.root = root.resolve()
        self.file_connections: Dict[Path, Set[Path]] = defaultdict(set)
        self.summaries: Dict[Path, FileSummary] = {}
        self.package_names: Set[str] = set()

        # Initialize concrete analyzers from registry
        self._analyzers = [cls() for cls in REGISTRY]
        self._supported_extensions = {
            ext for a in self._analyzers for ext in a.handled_extensions
        }

        self.logger = logger.bind(component="ProjectAnalyzer")
        self.logger.info("Initialized project analyzer",
                         root=str(self.root),
                         supported_extensions=list(self._supported_extensions))

    def identify_packages(self) -> Set[str]:
        """
        Identify all Python packages in the project.

        Returns:
            Set of top-level package names
        """
        self.logger.info("Identifying Python packages")

        for root, dirs, files in os.walk(self.root):
            if "__init__.py" in files:
                # This is a package
                package_path = Path(root).relative_to(self.root)
                top_level_package = package_path.parts[0] if package_path.parts else package_path.name
                self.package_names.add(top_level_package)
                self.logger.debug("Found package",
                                  package=top_level_package,
                                  path=str(package_path))

        # Update Python analyzers with package information for better import resolution
        for analyzer in self._analyzers:
            if isinstance(analyzer, REGISTRY[0].__class__):  # PythonAnalyzer
                try:
                    analyzer.set_package_names(self.package_names)
                except AttributeError:
                    pass  # Analyzer doesn't support package names

        return self.package_names

    def walk(self) -> Dict[Path, Set[Path]]:
        """
        Walk through the project directory and analyze files.

        Returns:
            Dictionary mapping files to their dependencies
        """
        self.logger.info("Walking project directory")

        # First identify packages to help with import resolution
        self.identify_packages()

        # Find all supported files
        paths = [
            p for p in self.root.rglob("*")
            if p.suffix in self._supported_extensions and p.is_file()
        ]
        self.logger.info(f"Found {len(paths)} files to analyze")

        # Analyze files in parallel
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(self._dispatch, p): p for p in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    deps = future.result()
                    self.file_connections[path.relative_to(self.root)].update(deps)
                except Exception as e:
                    self.logger.error("Error analyzing file",
                                      file=str(path.relative_to(self.root)),
                                      error=str(e))

        self.logger.info("Completed file analysis",
                         files_analyzed=len(self.file_connections))

        return self.file_connections

    def _dispatch(self, path: Path) -> Set[Path]:
        """
        Dispatch a file to the appropriate analyzer.

        Args:
            path: Path to the file to analyze

        Returns:
            Set of dependencies found by the analyzer
        """
        # Find the analyzer that can handle this file extension
        analyzer = next(
            (a for a in self._analyzers if path.suffix in a.handled_extensions),
            None
        )

        if not analyzer:
            self.logger.warning("No analyzer found for file",
                                file=str(path.relative_to(self.root)),
                                suffix=path.suffix)
            return set()

        # Analyze the file
        return analyzer.analyze(path, self.root)

    def generate_file_summaries(self) -> Dict[Path, FileSummary]:
        """
        Generate summaries for each analyzed file.

        Returns:
            Dictionary mapping file paths to their summaries
        """
        self.logger.info("Generating file summaries")

        # Process all files in the connections dictionary
        for rel_path, deps in self.file_connections.items():
            full_path = self.root / rel_path

            try:
                # Get basic file stats
                size = full_path.stat().st_size

                # Initialize summary with basic information
                summary = FileSummary(
                    path=rel_path,
                    language=rel_path.suffix.lstrip("."),
                    deps=list(deps),
                    size=size
                )

                # Extract language-specific information
                if rel_path.suffix == ".py":
                    self._enhance_python_summary(full_path, summary)
                elif rel_path.suffix in (".js", ".mjs", ".cjs"):
                    self._enhance_js_summary(full_path, summary)
                elif rel_path.suffix in (".md", ".markdown"):
                    self._enhance_markdown_summary(full_path, summary)

                # Add line count if file can be read as text
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        summary.line_count = sum(1 for _ in f)
                except Exception:
                    pass

                self.summaries[rel_path] = summary

            except Exception as e:
                self.logger.error("Error generating summary",
                                  file=str(rel_path),
                                  error=str(e))

        self.logger.info(f"Generated {len(self.summaries)} file summaries")

        return self.summaries

    def _enhance_python_summary(self, file_path: Path, summary: FileSummary) -> None:
        """
        Enhance a file summary with Python-specific information.

        Args:
            file_path: Path to the Python file
            summary: FileSummary to enhance
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Extract functions and classes
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

            summary.functions = functions[:10]  # Limit to top 10
            summary.classes = classes[:10]  # Limit to top 10

        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.warning("Error parsing Python file",
                                file=str(file_path),
                                error=str(e))

    def _enhance_js_summary(self, file_path: Path, summary: FileSummary) -> None:
        """
        Enhance a file summary with JavaScript-specific information.

        Args:
            file_path: Path to the JavaScript file
            summary: FileSummary to enhance
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Simple regex-based extraction
            functions = re.findall(r'function\s+(\w+)\s*\(', content)
            functions.extend(re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(', content))

            classes = re.findall(r'class\s+(\w+)', content)

            summary.functions = functions[:10]  # Limit to top 10
            summary.classes = classes[:10]  # Limit to top 10

        except Exception as e:
            self.logger.warning("Error analyzing JavaScript file",
                                file=str(file_path),
                                error=str(e))

    def _enhance_markdown_summary(self, file_path: Path, summary: FileSummary) -> None:
        """
        Enhance a file summary with Markdown-specific information.

        Args:
            file_path: Path to the Markdown file
            summary: FileSummary to enhance
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Extract headers
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)

            summary.headers = [h[1] for h in headers[:10]]  # Limit to top 10

        except Exception as e:
            self.logger.warning("Error analyzing Markdown file",
                                file=str(file_path),
                                error=str(e))

    def generate_openai_embeddings(self, api_key: str) -> None:
        """
        Generate OpenAI embeddings for file summaries.

        Args:
            api_key: OpenAI API key
        """
        self.logger.info("Generating OpenAI embeddings for file summaries")

        if not self.summaries:
            self.logger.warning("No file summaries to generate embeddings for")
            return

        for rel_path, summary in self.summaries.items():
            try:
                # Generate embedding from the summary text
                embedding_text = summary.get_embedding_text()
                summary.embedding = embed(embedding_text, api_key)
                self.logger.debug("Generated embedding", file=str(rel_path))
            except Exception as e:
                self.logger.error("Error generating embedding",
                                  file=str(rel_path),
                                  error=str(e))

        self.logger.info("Completed generating embeddings")

    def build_report(self) -> Dict[str, Any]:
        """
        Build a comprehensive report of the project analysis.

        Returns:
            Dictionary representation of the report
        """
        self.logger.info("Building project report")

        # Generate context summary if we have file connections
        context_summary = None
        if self.file_connections:
            # Convert paths to strings for the context summarizer
            string_connections = {
                str(k): {str(v) for v in vals}
                for k, vals in self.file_connections.items()
            }
            summarizer = ContextSummarizer(string_connections)
            context_summary = summarizer.generate_summary()

        # Generate directory tree
        tree = DirectoryTree(self.root)
        directory_tree = tree.generate_tree()

        # Build the report
        report = Report(
            root=str(self.root),
            interconnections={
                str(k): [str(d) for d in v]
                for k, v in self.file_connections.items()
            },
            summaries=list(self.summaries.values()),
            context_summary=context_summary,
            directory_tree=directory_tree
        )

        return report.to_dict()
import ast
import sys
import os
from pathlib import Path
from typing import Set, List

from .base import FileAnalyzer
from . import register
import structlog

logger = structlog.get_logger(__name__)


@register
class PythonAnalyzer(FileAnalyzer):
    """
    Analyzer for Python files that extracts imports and module dependencies.
    """
    handled_extensions = {".py"}

    def __init__(self):
        self.logger = logger.bind(analyzer="PythonAnalyzer")
        self.package_names: Set[str] = set()

    def set_package_names(self, package_names: Set[str]):
        """
        Set known package names for better import resolution.

        Args:
            package_names: Set of top-level package names in the project
        """
        self.package_names = package_names

    def analyze(self, file_path: Path, project_root: Path) -> Set[Path]:
        """
        Analyze a Python file to extract its import dependencies.

        Args:
            file_path: Path to the Python file
            project_root: Path to the project root directory

        Returns:
            Set of relative paths (from project_root) that this file depends on
        """
        deps: Set[Path] = set()
        rel_path = file_path.relative_to(project_root)

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
            self.logger.debug("Successfully parsed Python file", file=str(rel_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            self.logger.warning("Failed to parse Python file",
                                file=str(rel_path),
                                error=str(e))
            return deps

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self._add_import(project_root, deps, name.name, rel_path)
            elif isinstance(node, ast.ImportFrom) and node.module:
                # Handle relative imports with dots
                if node.level > 0:
                    # Get the current directory
                    current_dir = rel_path.parent
                    # Go up by level-1 directories (level 1 means current dir)
                    for _ in range(node.level - 1):
                        if current_dir.name:  # Ensure not at root
                            current_dir = current_dir.parent

                    # Combine with module if provided
                    module_path = current_dir
                    if node.module:
                        module_path = current_dir / Path(node.module.replace(".", "/"))

                    # Try to find the module
                    self._resolve_module_path(project_root, deps, module_path)
                else:
                    # Standard import from
                    self._add_import(project_root, deps, node.module, rel_path)

        return deps

    def _add_import(self, project_root: Path, deps: Set[Path], import_name: str, source_file: Path) -> None:
        """
        Add an import relationship to the dependencies set.

        Args:
            project_root: Project root path
            deps: Set to add dependencies to
            import_name: Name of the imported module
            source_file: Source file path that contains the import
        """
        # Skip standard library imports
        first_part = import_name.split(".")[0]
        if first_part in sys.builtin_module_names:
            return

        # Skip if it's not a known project package (external dependency)
        if first_part not in self.package_names and len(self.package_names) > 0:
            # Try direct file match as fallback
            direct_match = project_root / (first_part + ".py")
            if direct_match.exists():
                deps.add(direct_match.relative_to(project_root))
                self.logger.debug("Found direct file import",
                                  source=str(source_file),
                                  target=f"{first_part}.py")
            return

        # Try to resolve module path to a file
        module_path = Path(import_name.replace(".", "/"))
        self._resolve_module_path(project_root, deps, module_path)

    def _resolve_module_path(self, project_root: Path, deps: Set[Path], module_path: Path) -> None:
        """
        Resolve a module path to actual files and add to dependencies.

        Args:
            project_root: Project root path
            deps: Set to add dependencies to
            module_path: Path representation of the module
        """
        # Try multiple possible paths for the import
        candidates = [
            module_path.with_suffix(".py"),  # module.py
            module_path / "__init__.py",  # module/__init__.py
        ]

        for candidate in candidates:
            full_path = project_root / candidate
            if full_path.exists():
                deps.add(candidate)
                self.logger.debug("Resolved import", path=str(candidate))
                break
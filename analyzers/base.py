from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set


class FileAnalyzer(ABC):
    """
    Abstract base class for language-specific file analyzers.

    Each concrete analyzer implementation should:
    1. Define which file extensions it handles
    2. Implement the analyze method to extract dependencies
    """

    # Class variable to be defined by subclasses
    handled_extensions: Set[str] = set()

    @abstractmethod
    def analyze(self, file_path: Path, project_root: Path) -> Set[Path]:
        """
        Analyze a file to find its dependencies.

        Args:
            file_path: Path to the file being analyzed
            project_root: Path to the project root

        Returns:
            Set of relative paths (from project_root) of files this file depends on
        """
        pass
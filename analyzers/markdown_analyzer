import re
from pathlib import Path
from typing import Set

from .base import FileAnalyzer
from . import register
import structlog

logger = structlog.get_logger(__name__)

# Regular expression to match Markdown links does not include http links and anchors
MD_LINK_REGEX = re.compile(r"\[.*?\]\(((?!http)(?!#)[^)]+)\)")


@register
class MarkdownAnalyzer(FileAnalyzer):
    """
    Analyzer for Markdown files that extracts links to other files.
    """
    handled_extensions = {".md", ".markdown"}

    def __init__(self):
        self.logger = logger.bind(analyzer="MarkdownAnalyzer")

    def analyze(self, file_path: Path, project_root: Path) -> Set[Path]:
        """
        Analyze a Markdown file to extract links to other files.

        Args:
            file_path: Path to the Markdown file
            project_root: Path to the project root directory

        Returns:
            Set of relative paths (from project_root) that this file links to
        """
        deps: Set[Path] = set()
        rel_path = file_path.relative_to(project_root)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            self.logger.debug("Read Markdown file", file=str(rel_path))
        except Exception as e:
            self.logger.warning("Failed to read Markdown file",
                                file=str(rel_path),
                                error=str(e))
            return deps

        # Find all links in the content
        for link in MD_LINK_REGEX.findall(content):
            try:
                # Convert the link to a path (handle relative links)
                link_path = (file_path.parent / link).resolve()

                # Only include links to files that exist within the project
                if link_path.exists() and project_root in link_path.parents:
                    deps.add(link_path.relative_to(project_root))
                    self.logger.debug("Found Markdown link",
                                      source=str(rel_path),
                                      target=str(link_path.relative_to(project_root)))
            except (ValueError, RuntimeError) as e:
                self.logger.warning("Error resolving Markdown link",
                                    link=link,
                                    error=str(e))

        return deps
import re
from pathlib import Path
from typing import Set

from .base import FileAnalyzer
from . import register
import structlog

logger = structlog.get_logger(__name__)

# expression that match JavaScript imports
JS_IMPORT_REGEX = re.compile(
    r"""(?:import\s+.+?\s+from\s+|require\()\s*["']([^"']+)["']""",
    re.MULTILINE
)


@register
class JSAnalyzer(FileAnalyzer):
    """
    Analyzer for JavaScript files that extracts imports and dependencies.
    """
    handled_extensions = {".js", ".mjs", ".cjs"}

    def __init__(self):
        self.logger = logger.bind(analyzer="JSAnalyzer")

    def analyze(self, file_path: Path, project_root: Path) -> Set[Path]:
        """
        Analyze a JavaScript file to extract its import dependencies.

        Args:
            file_path: Path to the JavaScript file
            project_root: Path to the project root directory

        Returns:
            Set of relative paths (from project_root) that this file depends on
        """
        deps: Set[Path] = set()
        rel_path = file_path.relative_to(project_root)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            self.logger.debug("Read JavaScript file", file=str(rel_path))
        except Exception as e:
            self.logger.warning("Failed to read JavaScript file",
                                file=str(rel_path),
                                error=str(e))
            return deps

        # Find all imports in the content
        for match in JS_IMPORT_REGEX.findall(content):
            # Only handle relative imports (skip npm packages)
            if match.startswith((".", "/")):
                try:
                    # Normalize the path
                    if match.startswith("/"):
                        # Absolute path within project
                        import_path = Path(match.lstrip("/"))
                    else:
                        # Relative path
                        import_path = (file_path.parent / match).resolve().relative_to(project_root)

                    # Try different extensions if none provided
                    if not import_path.suffix:
                        for ext in [".js", ".mjs", ".cjs"]:
                            test_path = import_path.with_suffix(ext)
                            if (project_root / test_path).exists():
                                deps.add(test_path)
                                self.logger.debug("Found JS import",
                                                  source=str(rel_path),
                                                  target=str(test_path))
                                break
                    else:
                        # Extension provided
                        if (project_root / import_path).exists():
                            deps.add(import_path)
                            self.logger.debug("Found JS import",
                                              source=str(rel_path),
                                              target=str(import_path))

                except (ValueError, RuntimeError) as e:
                    self.logger.warning("Error resolving JS import",
                                        import_path=match,
                                        error=str(e))

        return deps
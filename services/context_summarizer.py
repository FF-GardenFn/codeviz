from typing import Dict, List, Set, Any
from collections import defaultdict, Counter
import structlog
from ..models.report import ContextSummary

logger = structlog.get_logger(__name__)


class ContextSummarizer:
    """
    Generates high-level context summaries from project analysis data.
    """

    def __init__(self, file_connections: Dict[str, Set[str]]):
        """
        Initialize with file connections data.

        Args:
            file_connections: Dictionary mapping source files to their dependencies
        """
        self.file_connections = file_connections
        self.logger = logger.bind(component="ContextSummarizer")

    def generate_summary(self, max_items: int = 10) -> ContextSummary:
        """
        Generate a high-level context summary of the project.

        Args:
            max_items: Maximum number of items to include in each category

        Returns:
            ContextSummary object with the generated summary
        """
        self.logger.info("Generating context summary")

        # Find the most connected files (those with most dependencies)
        most_connected = sorted(
            self.file_connections.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:max_items]

        # Find the most commonly imported modules
        all_imports = []
        for deps in self.file_connections.values():
            all_imports.extend(deps)

        import_counter = Counter(all_imports)
        top_imports = import_counter.most_common(max_items)

        # Generate import chains (files that import files that import files...)
        import_chains = []
        for source, deps in self.file_connections.items():
            for dep in deps:
                if dep in self.file_connections and self.file_connections[dep]:
                    # This dependency imports other files
                    for second_dep in self.file_connections[dep]:
                        import_chain = f"{source} → {dep} → {second_dep}"
                        import_chains.append(import_chain)

        # Take top import chains for summary
        import_chains = import_chains[:max_items]

        # Calculate total connections
        total_connections = sum(len(deps) for deps in self.file_connections.values())

        # Create and return the context summary
        return ContextSummary(
            most_connected_files=[
                {"file": str(src), "connections": len(deps)}
                for src, deps in most_connected
            ],
            top_imported_modules=[
                {"module": str(mod), "imported_by_count": count}
                for mod, count in top_imports
            ],
            import_chains=import_chains,
            total_files_analyzed=len(self.file_connections),
            total_connections=total_connections
        )
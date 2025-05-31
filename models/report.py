from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from .file_summary import FileSummary


class ContextSummary(BaseModel):
    """
    High-level summary of the project context and relationships.
    """
    most_connected_files: List[Dict[str, Any]] = Field(default_factory=list)
    top_imported_modules: List[Dict[str, Any]] = Field(default_factory=list)
    import_chains: List[str] = Field(default_factory=list)
    total_files_analyzed: int = 0
    total_connections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        return {
            "most_connected_files": self.most_connected_files,
            "top_imported_modules": self.top_imported_modules,
            "import_chains": self.import_chains,
            "total_files_analyzed": self.total_files_analyzed,
            "total_connections": self.total_connections,
        }


class Report(BaseModel):
    """
    Complete report of the project analysis.
    """
    root: str
    interconnections: Dict[str, List[str]]
    summaries: List[FileSummary] = Field(default_factory=list)
    context_summary: Optional[ContextSummary] = None
    directory_tree: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        result = {
            "root": self.root,
            "interconnections": self.interconnections,
            "summaries": [s.to_dict() for s in self.summaries],
        }

        if self.context_summary:
            result["context_summary"] = self.context_summary.to_dict()

        if self.directory_tree:
            result["directory_tree"] = self.directory_tree

        return result
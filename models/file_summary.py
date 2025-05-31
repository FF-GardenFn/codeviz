from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FileSummary(BaseModel):
    """
    Model representing a summary of a file in the codebase.
    Contains metadata about the file's structure and content.
    """
    path: Path
    language: str
    functions: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    deps: List[Path] = Field(default_factory=list)
    size: int
    embedding: Optional[List[float]] = None

    # Additional metadata for specific file types
    headers: Optional[List[str]] = None  # For markdown files
    line_count: Optional[int] = None

    def get_embedding_text(self) -> str:
        """
        Generate a text representation of this file summary
        suitable for embedding generation.
        """
        components = [
            f"Path: {self.path}",
            f"Language: {self.language}",
            f"Size: {self.size} bytes",
        ]

        if self.functions:
            components.append(f"Functions: {', '.join(self.functions)}")

        if self.classes:
            components.append(f"Classes: {', '.join(self.classes)}")

        if self.headers:
            components.append(f"Headers: {', '.join(self.headers)}")

        if self.deps:
            components.append(f"Dependencies: {', '.join(str(d) for d in self.deps)}")

        return "\n".join(components)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for JSON serialization."""
        result = {
            "path": str(self.path),
            "language": self.language,
            "functions": self.functions,
            "classes": self.classes,
            "dependencies": [str(d) for d in self.deps],
            "size": self.size,
        }

        if self.headers:
            result["headers"] = self.headers

        if self.line_count:
            result["line_count"] = self.line_count

        if self.embedding:
            result["has_embedding"] = True
            # Don't include actual embedding in output - too verbose

        return result
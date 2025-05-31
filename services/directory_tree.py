from pathlib import Path
import os


class DirectoryTree:
    """
    Generates a visual representation of a directory structure.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.tee = "├── "
        self.last = "└── "
        self.branch = "│   "
        self.indent = "    "

    def generate_tree(self) -> str:
        """
        Generate the directory tree structure as a string.
        """
        lines = [str(self.root)]
        self._walk(self.root, "", lines)
        return "\n".join(lines)

    def _walk(self, current: Path, prefix: str, out: list[str]) -> None:
        """
        Recursively walk through directories and files, building tree representation.

        Args:
            current: The current path to explore
            prefix: Current line prefix for tree formatting
            out: List of output lines to append to
        """
        # Sort entries case-insensitively and filter hidden files/dirs
        entries = [
            e for e in sorted(current.iterdir(), key=lambda p: p.name.lower())
            if not e.name.startswith(".")
        ]

        # Process each entry
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = self.last if is_last else self.tee

            # Add entry to tree
            out.append(f"{prefix}{connector}{entry.name}")

            # Recursively process directories
            if entry.is_dir():
                next_prefix = prefix + (self.indent if is_last else self.branch)
                self._walk(entry, next_prefix, out)
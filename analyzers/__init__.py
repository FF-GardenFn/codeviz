from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Type, List
from .base import FileAnalyzer  # noqa – needed for typing

REGISTRY: List[Type[FileAnalyzer]] = []

def register(cls: Type[FileAnalyzer]) -> Type[FileAnalyzer]:
    """
    Decorator to add concrete analyzer classes to the global registry.
    """
    REGISTRY.append(cls)
    return cls

# Auto-import sibling analyzer modules so they self-register
for _file in Path(__file__).parent.glob("*.py"):
    if _file.stem not in {"__init__", "base", "project_analyzer"}:
        _module_path = f"{__package__}.{_file.stem}"
        try:
            import_module(_module_path)  # side-effect: register()
        except ImportError as e:
            print(f"Warning: Could not import analyzer module {_module_path}: {e}")

from .project_analyzer import ProjectAnalyzer

__all__ = ["FileAnalyzer", "ProjectAnalyzer", "register", "REGISTRY"]
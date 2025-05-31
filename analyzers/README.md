# CodeViz Analyzers

The Analyzers module is responsible for analyzing code files and project structure to extract meaningful information about your codebase.

## Overview

The Analyzers module contains a set of specialized analyzers for different file types and a project analyzer that orchestrates the analysis process. Each analyzer is designed to extract relevant information from specific file types, such as Python, JavaScript, or Markdown.

## Components

### Base Analyzer

`base.py` defines the `FileAnalyzer` base class that all concrete analyzers must implement. It provides the interface for analyzing files and extracting dependencies.

```python
class FileAnalyzer:
    """Base class for file analyzers."""
    
    handled_extensions: List[str] = []
    
    def analyze(self, path: Path, root: Path) -> Set[Path]:
        """
        Analyze a file and extract its dependencies.
        
        Args:
            path: Path to the file to analyze
            root: Path to the project root
            
        Returns:
            Set of dependencies found in the file
        """
        raise NotImplementedError("Concrete analyzers must implement this method")
```

### Python Analyzer

`python_analyzer.py` analyzes Python files to extract imports, classes, functions, and other relevant information.

Features:
- Detects Python imports and resolves them to file paths
- Identifies classes and functions
- Handles relative imports
- Supports package-based imports

### JavaScript Analyzer

`js_analyzer.py` analyzes JavaScript files to extract imports, exports, classes, functions, and other relevant information.

Features:
- Detects ES6 imports and CommonJS requires
- Identifies classes, functions, and variables
- Handles relative imports
- Supports module exports

### Markdown Analyzer

`markdown_analyzer.py` analyzes Markdown files to extract headers, links, and other relevant information.

Features:
- Extracts headers and their hierarchy
- Identifies links to other files
- Detects code blocks and their language

### Project Analyzer

`project_analyzer.py` orchestrates the analysis process by:
1. Walking through the project directory
2. Dispatching files to appropriate analyzers
3. Collecting and aggregating results
4. Generating file summaries
5. Creating OpenAI embeddings
6. Building a comprehensive report

## Usage

The Analyzers module is primarily used through the `ProjectAnalyzer` class:

```python
from pathlib import Path
from codeviz.analyzers.project_analyzer import ProjectAnalyzer

# Initialize the project analyzer
analyzer = ProjectAnalyzer(Path("./my_project"))

# Walk the project to find dependencies
analyzer.walk()

# Generate file summaries
analyzer.generate_file_summaries()

# Generate OpenAI embeddings (requires API key)
analyzer.generate_openai_embeddings("your-api-key")

# Build a comprehensive report
report = analyzer.build_report()
```

## Extension

To add support for a new file type, create a new analyzer class that inherits from `FileAnalyzer` and implements the `analyze` method. Then, register it using the `@register` decorator:

```python
from pathlib import Path
from typing import Set
from codeviz.analyzers import register
from codeviz.analyzers.base import FileAnalyzer

@register
class RustAnalyzer(FileAnalyzer):
    """Analyzer for Rust files."""
    
    handled_extensions = [".rs"]
    
    def analyze(self, path: Path, root: Path) -> Set[Path]:
        # Implementation for analyzing Rust files
        dependencies = set()
        # ... extract dependencies ...
        return dependencies
```

The  analyzer will be automatically discovered and used by the `ProjectAnalyzer`.
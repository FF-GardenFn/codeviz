# CodeViz Services

The Services module provides supporting functionality for the CodeViz tool, including context summarization, directory tree visualization, and OpenAI embeddings.

## Overview

The Services module contains a set of utility services that support the core functionality of CodeViz. These services handle tasks such as summarizing context information, generating directory tree visualizations, and creating OpenAI embeddings.

## Components

### Context Summarizer

`context_summarizer.py` analyzes the interconnections between files in a project to generate high-level insights about the project structure.

Features:
- Identifies the most connected files in the project
- Discovers top imported modules
- Traces import chains to understand dependencies
- Generates a comprehensive context summary

Example usage:
```python
from codeviz.services.context_summarizer import ContextSummarizer

# Create a context summarizer with file connections
connections = {
    "file1.py": {"file2.py", "file3.py"},
    "file2.py": {"file3.py"},
    "file3.py": set()
}
summarizer = ContextSummarizer(connections)

# Generate a context summary
summary = summarizer.generate_summary()

# Access summary information
most_connected = summary.most_connected_files
top_modules = summary.top_imported_modules
import_chains = summary.import_chains
```

### Directory Tree

`directory_tree.py` generates a visual representation of a directory structure, making it easier to understand the organization of a project.

Features:
- Creates ASCII tree representation of directory structure
- Supports customization of indentation and branch characters
- Can exclude specific directories or files
- Handles large directory structures efficiently

Example usage:
```python
from pathlib import Path
from codeviz.services.directory_tree import DirectoryTree

# Create a directory tree for a project
tree = DirectoryTree(Path("./my_project"))

# Generate the tree representation
tree_str = tree.generate_tree()

# Print the tree
print(tree_str)
```

### OpenAI Embeddings

`openai_embeddings.py` provides functionality for generating and caching OpenAI embeddings, which are used for semantic similarity analysis.

Features:
- Generates embeddings using the OpenAI API
- Caches embeddings to avoid redundant API calls
- Handles rate limiting and retries
- Provides a clean interface for embedding generation

Example usage:
```python
from code_awac.services.openai_embeddings import embed, clear_cache

# Generate an embedding for a text
api_key = "your-openai-api-key"
text = "This is a sample text to embed"
embedding = embed(text, api_key)

# Use the embedding for similarity analysis
# ...

# Clear the embedding cache if needed
num_cleared = clear_cache()
```

## Integration with Other Modules

The Services module is used by both the Analyzers and Discharge modules:

- **Analyzers**: The Project Analyzer uses the Context Summarizer to generate insights about the project structure and the OpenAI Embeddings service to create embeddings for file summaries.

- **Discharge**: The Context Bridge uses the OpenAI Embeddings service to generate embeddings for chat messages and code files.

## Extension

To add a new service, create a new Python file in the services directory with your service implementation. Then, import and use it in your code as needed.

For example, to add a new service for syntax highlighting (this is a hypothetical example, not an actual file in the codebase):

```python
# Example: codeviz/services/syntax_highlighter.py
class SyntaxHighlighter:
    """Service for syntax highlighting code snippets."""

    def __init__(self, theme="default"):
        self.theme = theme

    def highlight(self, code: str, language: str) -> str:
        """
        Highlight code in the specified language.

        Args:
            code: The code to highlight
            language: The programming language

        Returns:
            Highlighted code as HTML
        """
        # Implementation of your syntax highlighting
        # ...
        return highlighted_code
```

Then you could use it in your code like this:

```python
# Example usage of the hypothetical syntax highlighter
from codeviz.services.syntax_highlighter import SyntaxHighlighter

highlighter = SyntaxHighlighter(theme="monokai")
highlighted_code = highlighter.highlight("print('Hello, world!')", "python")
```

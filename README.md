![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Last Commit](https://img.shields.io/github/last-commit/FF-GardenFn/codeviz)

# CodeViz - Comprehensive Code Analysis and Context Bridging Tool

CodeViz is a powerful code analysis and context bridging tool that helps you understand your codebase by analyzing dependencies, generating summaries, creating visualizations enhanced with OpenAI embeddings, and bridging code with conversation context.

## 🌟 Features

### Code Analysis
* 📊 **Dependency Analysis**: Identifies connections between files in your codebase
* 🔍 **Context Summaries**: Generates high-level insights about your project structure
* 📝 **File Summaries**: Creates detailed summaries of each file's contents
* 🧠 **OpenAI Embeddings**: Generates semantic embeddings for advanced similarity analysis
* 🌲 **Directory Visualization**: Displays the directory structure in an easy-to-read format
* 🧩 **Multi-language Support**: Analyzes Python, JavaScript, and Markdown files

### Semantic Analysis
* 🔬 **Similarity Analysis**: Finds semantically similar files in your codebase
* 👥 **Clustering**: Identifies clusters of related files
* 🔄 **Refactoring Suggestions**: Provides suggestions for code consolidation and improvement
* 📝 **LLM-Ready Prompts**: Converts similarity analysis into markdown prompts for LLMs

### Context Bridging
* 🔗 **Chat Context Integration**: Connects conversation history with relevant code
* 💬 **Chat Extraction**: Extracts conversations from AI assistants like Claude, ChatGPT, and others
* 📝 **Enhanced Prompts**: Generates context-rich prompts that incorporate code and conversation
* 🧪 **Code Relevance**: Identifies code files relevant to specific conversation points

## 📋 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Commands](#commands)
  - [analyse](#analyse)
  - [tree](#tree)
  - [bridge](#bridge)
  - [analyze_embeddings](#analyze_embeddings)
  - [similarity_to_prompt](#similarity_to_prompt)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Installation Steps

```bash
# Clone the repository 
git clone https://github.com/FF-GardenFn/codeviz
cd codeviz

# Install the package
pip install -e .
```

For development:
```bash
# Install development dependencies
pip install -e ".[dev]"
```

## 🏁 Quick Start

### Basic Code Analysis
```bash
# Analyze current directory
codeviz analyse .

# Generate comprehensive report with all features
codeviz analyse . --tree --summary --context --embeddings

# Output to specific file
codeviz analyse . -o my-report.json
```

### Directory Visualization
```bash
# Display the directory tree
codeviz tree
```

### Context Bridging
```bash
# Bridge chat context with codebase
codeviz bridge chat_export.json ./my_project --output enhanced_prompt.md

# Print the generated prompt to console
codeviz bridge chat_export.json ./my_project --print
```

### Embedding Analysis
```bash
# Analyze embeddings from a report
codeviz analyze_embeddings codeviz-report.json

# Adjust similarity threshold and top-k similar files
codeviz analyze_embeddings codeviz-report.json --threshold 0.75 --top-k 10

# Convert similarity report to LLM-ready prompt
codeviz similarity_to_prompt similarity-report.json

# Customize the prompt output
codeviz similarity_to_prompt similarity-report.json --max-files-cluster 5 --max-similar 3 --print
```

## ⚙️ Configuration

### OpenAI API Key

For embedding generation and context bridging, CodeViz requires an OpenAI API key. You can provide it in several ways:

1. Environment variable:
   ```bash
   export CODEVIZ_OPENAI_API_KEY="sk-..."
   ```

2. In a `.env` file in your project directory:
   ```
   CODEVIZ_OPENAI_API_KEY=sk-...
   ```

3. Command line argument:
   ```bash
   codeviz analyse . --embeddings --api-key sk-...
   codeviz bridge chat.json . --api-key sk-...
   ```

## 🛠️ Commands

### analyse

Analyze a project directory and generate a comprehensive report.

```bash
codeviz analyse [OPTIONS] [ROOT]
```

#### Arguments
- `ROOT`: Project root directory to analyze (default: current directory)

#### Options
- `-o, --out PATH`: Path to write JSON report (default: codeviz-report.json)
- `-t, --tree`: Print directory tree
- `-s, --summary`: Generate per-file summaries
- `-c, --context`: Generate project context summary
- `-e, --embeddings`: Generate OpenAI embeddings
- `--api-key TEXT`: OpenAI API key (overrides environment variable)

### tree

Generate and display a directory tree.

```bash
codeviz tree [OPTIONS] [PATH]
```

#### Arguments
- `PATH`: Directory to display tree for (default: current directory)

### bridge

Generate enhanced prompts with chat context and relevant code.

```bash
codeviz bridge [OPTIONS] CHAT CODEBASE
```

#### Arguments
- `CHAT`: Path to chat export file
- `CODEBASE`: Path to codebase directory (default: current directory)

#### Options
- `-o, --output PATH`: Output file for the enhanced prompt (default: enhanced_prompt.md)
- `-t, --tokens INTEGER`: Maximum tokens for the prompt (default: 3000)
- `--api-key TEXT`: OpenAI API key (uses OPENAI_API_KEY environment variable if not provided)
- `--threshold FLOAT`: Similarity threshold for code relevance (default: 0.7)
- `-d, --debug`: Enable debug logging
- `-p, --print`: Print the generated prompt to console

### analyze_embeddings

Analyze semantic similarity between files based on their embeddings.

```bash
codeviz analyze_embeddings [OPTIONS] REPORT
```

#### Arguments
- `REPORT`: Path to JSON file with embeddings

#### Options
- `-o, --output PATH`: Output JSON file for similarity results (default: similarity-report.json)
- `-k, --top-k INTEGER`: Number of top similar neighbors to report per file (default: 5)
- `-t, --threshold FLOAT`: Similarity threshold for clustering (default: 0.7)
- `-d, --debug`: Print debug information about the input file

### similarity_to_prompt

Convert a similarity report to a markdown prompt for use with LLMs.

```bash
codeviz similarity_to_prompt [OPTIONS] REPORT
```

#### Arguments
- `REPORT`: Path to similarity report JSON file

#### Options
- `-o, --output PATH`: Output file for the generated prompt (default: similarity-prompt.md)
- `-m, --max-files-cluster INTEGER`: Maximum number of files to show per cluster (default: 10)
- `-s, --max-similar INTEGER`: Maximum number of similar files to show per file (default: 5)
- `-p, --print`: Print the generated prompt to console

## 🏗️ Architecture

CodeViz is organized into several modules:

### analyzers
The `analyzers` module contains code for analyzing different types of files:
- `base.py`: Base class for analyzers
- `js_analyzer.py`: Analyzer for JavaScript files
- `markdown_analyzer.py`: Analyzer for Markdown files
- `project_analyzer.py`: Main analyzer for projects
- `python_analyzer.py`: Analyzer for Python files

### discharge
The `discharge` module provides tools for chat extraction and context bridging:
- `analyze_embeddings.py`: Analyzes embeddings for semantic similarity
- `chat_extract.js`: JavaScript tool for extracting chat content from AI assistants
- `chat_processor.py`: Processes chat data
- `code_scanner.py`: Scans and processes code files
- `context_bridge.py`: Bridges code similarity analysis with chat context
- `embeddings_utils.py`: Utilities for generating and working with embeddings

### services
The `services` module provides supporting functionality:
- `context_summarizer.py`: Summarizes context information
- `directory_tree.py`: Generates directory tree visualizations
- `openai_embeddings.py`: Handles OpenAI embeddings

### models
The `models` module contains data models used throughout the application.

## 💡 Use Cases

* 🔄 **Onboarding**: Help new developers understand project structure
* 🏗️ **Refactoring**: Identify dependencies before making changes
* 📚 **Documentation**: Generate project insights for documentation
* 🔎 **Code Review**: Understand how new code impacts existing structure
* 🤖 **AI Assistance**: Create context-rich prompts for AI assistants
* 🧠 **Knowledge Management**: Bridge conversations with relevant code
* 🔍 **Code Discovery**: Find semantically similar code across your project

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -m 'Add my feature'`
6. Push to the branch: `git push origin feature/my-feature`
7. Submit a pull request

Please make sure your code follows the project's coding style and includes appropriate tests.

## 📄 License

MIT License

Copyright (c) 2025 [Faycal Farhat](https://github.com/FF-GardenFn)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

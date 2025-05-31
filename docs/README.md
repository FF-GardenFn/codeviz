# CodeViz Documentation

Welcome to the CodeViz documentation! This directory contains comprehensive documentation for the CodeViz tool, a powerful code analysis and context bridging tool that helps you understand your codebase.

## Documentation Overview

The documentation is organized into several sections:

1. **Main README**: The [main README](../README.md) provides an overview of CodeViz, including its features, installation instructions, and basic usage examples.

2. **Module Documentation**:
   - [Analyzers](../analyzers/README.md): Documentation for the analyzers module, which is responsible for analyzing code files and project structure.
   - [Services](../services/README.md): Documentation for the services module, which provides supporting functionality for CodeViz.
   - [Discharge](../discharge/instructions.md): Documentation for the discharge module, which provides tools for chat extraction and context bridging.

3. **Usage Examples**: The [usage examples](usage_examples.md) document provides practical examples of how to use CodeViz for various scenarios, from basic code analysis to advanced context bridging.

## Getting Started

If you're new to CodeViz, we recommend starting with the [main README](../README.md) to get an overview of the tool and its capabilities. Then, check out the [usage examples](usage_examples.md) to see how to apply CodeViz to your specific needs.

For more detailed information about specific modules, refer to the module documentation:
- [Analyzers](../analyzers/README.md) for code analysis
- [Services](../services/README.md) for supporting functionality
- [Discharge](../discharge/instructions.md) for chat extraction and context bridging

## Command Reference

CodeViz provides several commands for different tasks:

- `codeviz analyse`: Analyze a project directory and generate a comprehensive report.
- `codeviz tree`: Generate and display a directory tree.
- `codeviz bridge`: Generate enhanced prompts with chat context and relevant code.
- `codeviz analyze_embeddings`: Analyze semantic similarity between files based on their embeddings.
- `codeviz similarity_to_prompt`: Convert a similarity report to a markdown prompt for use with LLMs.

For detailed information about each command and its options, refer to the [Commands](../README.md#commands) section in the main README.

## Use Cases

CodeViz can be used for various purposes, including:

- **Onboarding**: Help new developers understand project structure
- **Refactoring**: Identify dependencies before making changes
- **Documentation**: Generate project insights for documentation
- **Code Review**: Understand how new code impacts existing structure
- **AI Assistance**: Create context-rich prompts for AI assistants
- **Knowledge Management**: Bridge conversations with relevant code
- **Code Discovery**: Find semantically similar code across your project

For detailed examples of these use cases, refer to the [Advanced Scenarios](usage_examples.md#advanced-scenarios) section in the usage examples document.

## Contributing

If you'd like to contribute to CodeViz, please refer to the [Contributing](../README.md#contributing) section in the main README for guidelines on how to contribute.

## License

CodeViz is released under the MIT License. For more information, refer to the [License](../README.md#license) section in the main README.

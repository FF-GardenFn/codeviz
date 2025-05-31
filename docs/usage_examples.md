# CodeViz Usage Examples

This document provides practical examples of how to use CodeViz for various scenarios, from basic code analysis to advanced context bridging.

## Table of Contents
- [Basic Code Analysis](#basic-code-analysis)
- [Directory Visualization](#directory-visualization)
- [Semantic Similarity Analysis](#semantic-similarity-analysis)
- [Chat Context Bridging](#chat-context-bridging)
- [Advanced Scenarios](#advanced-scenarios)
  - [Refactoring Planning](#refactoring-planning)
  - [Onboarding New Developers](#onboarding-new-developers)
  - [AI-Assisted Development](#ai-assisted-development)

## Basic Code Analysis

### Analyzing a Project

To analyze a project and generate a basic report:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Run the analysis
codeviz analyse .
```

This will generate a `codeviz-report.json` file in the current directory with basic information about your project.

### Comprehensive Analysis

For a more comprehensive analysis that includes file summaries, context information, and embeddings:

```bash
codeviz analyse . --summary --context --embeddings --api-key your-openai-api-key
```

### Viewing Analysis Results

To view a summary of the analysis results directly in the terminal:

```bash
codeviz analyse . --summary --context --tree
```

This will display the directory tree and context summary in the terminal, along with generating the full report.

## Directory Visualization

### Generating a Directory Tree

To generate and display a directory tree for a project:

```bash
codeviz tree /path/to/your/project
```

### Saving the Directory Tree

To save the directory tree to a file:

```bash
codeviz tree /path/to/your/project > directory_tree.txt
```

## Semantic Similarity Analysis

### Analyzing Embeddings

After generating a report with embeddings, you can analyze the semantic similarity between files:

```bash
codeviz analyze_embeddings codeviz-report.json
```

### Customizing Similarity Analysis

You can customize the similarity analysis by adjusting the threshold and number of similar files:

```bash
codeviz analyze_embeddings codeviz-report.json --threshold 0.8 --top-k 10
```

### Saving Similarity Results

To save the similarity analysis results to a specific file:

```bash
codeviz analyze_embeddings codeviz-report.json --output similarity-report.json
```

### Converting Similarity Results to LLM Prompts

To convert similarity analysis results into a markdown prompt for use with LLMs:

```bash
codeviz similarity_to_prompt similarity-report.json
```

### Customizing the Prompt

You can customize the generated prompt by adjusting the number of files shown per cluster and the number of similar files shown per file:

```bash
codeviz similarity_to_prompt similarity-report.json --max-files-cluster 5 --max-similar 3
```

### Viewing the Generated Prompt

To view the generated prompt directly in the terminal:

```bash
codeviz similarity_to_prompt similarity-report.json --print
```

## Chat Context Bridging

### Extracting Chat Content

1. Install the AI Chat Extractor bookmarklet by creating a new bookmark in your browser and pasting the content of `chat_extract.js` as the URL.
2. Navigate to an AI assistant chat page (Claude, ChatGPT, etc.).
3. Click the bookmarklet to extract the conversation.
4. Click "Download JSON" to save the conversation to a file (e.g., `chat_export.json`).

### Bridging Chat with Code

To bridge the extracted chat with your codebase:

```bash
codeviz bridge chat_export.json /path/to/your/project --output enhanced_prompt.md
```

This will generate an enhanced prompt that includes relevant code snippets based on the conversation context.

### Customizing the Bridge

You can customize the bridge by adjusting the token limit and similarity threshold:

```bash
codeviz bridge chat_export.json /path/to/your/project --tokens 4000 --threshold 0.6
```

### Viewing the Enhanced Prompt

To view the enhanced prompt directly in the terminal:

```bash
codeviz bridge chat_export.json /path/to/your/project --print
```

## Advanced Scenarios

### Refactoring Planning

When planning a refactoring, you can use CodeViz to identify dependencies and similar code:

1. Analyze the project:
   ```bash
   codeviz analyse . --summary --context --embeddings
   ```

2. Analyze semantic similarities to find potential code duplication:
   ```bash
   codeviz analyze_embeddings codeviz-report.json --threshold 0.85
   ```

3. Review the refactoring suggestions in the similarity report to identify areas for consolidation.

### Onboarding New Developers

To help new developers understand a project:

1. Generate a comprehensive project report:
   ```bash
   codeviz analyse . --summary --context --tree
   ```

2. Create a directory visualization:
   ```bash
   codeviz tree . > project_structure.txt
   ```

3. Identify the most important files and their relationships from the context summary in the report.

4. Share these resources with new team members to help them navigate the codebase.

### AI-Assisted Development

To leverage AI assistants for development tasks:

1. Extract a conversation about a specific development task:
   - Use the AI Chat Extractor bookmarklet to extract the conversation.
   - Save the JSON file.

2. Bridge the conversation with your codebase:
   ```bash
   codeviz bridge chat_export.json . --output task_context.md
   ```

3. Use the enhanced prompt to continue the conversation with the AI assistant, now with relevant code context included.

4. Iterate on the development task, extracting new conversations and bridging them as needed.

This workflow helps maintain context between AI assistant sessions and ensures the assistant has access to the most relevant code for the task at hand.

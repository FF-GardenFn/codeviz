# AI Chat Extractor and Context Bridging Tools

The Discharge module provides tools for extracting chat content from AI assistants and bridging it with relevant code context.

## Table of Contents
- [AI Chat Extractor Bookmarklet](#ai-chat-extractor-bookmarklet)
- [Context Bridge](#context-bridge)
- [Embedding Analysis](#embedding-analysis)
- [Chat Processing](#chat-processing)
- [Code Scanning](#code-scanning)
- [Embedding Utilities](#embedding-utilities)

## AI Chat Extractor Bookmarklet

This bookmarklet allows you to extract conversations from various AI assistants (Claude, ChatGPT, Bard, etc.) directly in your browser.

### How to Install the Bookmarklet

1. Create a new bookmark in your browser
2. Name it "AI Chat Extractor"
3. Copy the entire content of the `chat_extract.js` file
4. Paste it as the URL of your bookmark

### How to Use

1. Navigate to any AI assistant chat page (Claude, ChatGPT, Bard, etc.)
2. Click on your "AI Chat Extractor" bookmark
3. A popup will appear in the page
4. Click "Extract Conversation" to extract the chat
5. Click "Copy to Clipboard" to copy the JSON or "Download JSON" to save it
6. Use the extracted JSON with the context bridge:

```bash
codeviz bridge chat_export.json ./my_project --output enhanced_prompt.md
```

### Features

- Works directly in your browser
- One-click extraction from various AI assistant interfaces
- Downloads JSON file or copies to clipboard
- Compatible with multiple AI assistants (Claude, ChatGPT, Bard, etc.)
- No server-side processing required
- Works with code blocks and formatting

### Troubleshooting

If extraction fails:
1. Try refreshing the page and trying again
2. Check if the page has fully loaded
3. If all else fails, use the manual copy-paste method

### Note on Security

This bookmarklet runs only in your browser and doesn't send data to any external servers. All processing happens locally in your browser.

## Context Bridge

The Context Bridge connects chat conversations with relevant code to create enhanced prompts for AI assistants.

### Usage

```bash
codeviz bridge chat_export.json ./my_project --output enhanced_prompt.md
```

### Features

- Analyzes chat messages to identify key topics
- Scans codebase to find relevant files
- Generates embeddings for semantic similarity matching
- Creates enhanced prompts that include both conversation context and relevant code
- Supports customization of token limits and similarity thresholds

## Embedding Analysis

The Embedding Analysis tool finds semantically similar files in your codebase and can convert the analysis into LLM-ready prompts.

### Usage

```bash
# Analyze embeddings
codeviz analyze_embeddings codeviz-report.json

# Convert similarity report to LLM prompt
codeviz similarity_to_prompt similarity-report.json
```

### Features

- Identifies similar files based on semantic embeddings
- Finds clusters of related files
- Provides refactoring suggestions
- Supports customization of similarity thresholds and number of neighbors
- Converts similarity analysis into markdown prompts for LLMs
- Customizable prompt generation with control over detail level

## Chat Processing

The Chat Processing module handles loading, parsing, and processing chat data from various formats.

### Features

- Supports multiple chat export formats (JSON, plain text)
- Extracts messages with proper role attribution
- Clusters and selects important messages
- Generates excerpts from code files

## Code Scanning

The Code Scanner module scans and processes code files from a codebase.

### Features

- Supports multiple file types
- Handles large codebases efficiently
- Provides file content for analysis
- Finds code relevant to specific chat messages

## Embedding Utilities

The Embedding Utilities module provides tools for generating and working with embeddings.

### Features

- Generates embeddings using OpenAI API
- Provides fallback embedding generation when API is unavailable
- Computes similarity between embeddings
- Estimates token counts for text

#!/usr/bin/env python3
"""
context_bridge.py

Bridges code similarity analysis with chat context distillation to create
rich, contextually-aware prompts that incorporate both conversation history
and relevant code.

Usage:
  python context_bridge.py --chat chat_export.json --codebase ./my_project --output enhanced_prompt.md
"""
import json
import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from datetime import datetime
import time


from .embeddings_utils import get_embedding, compute_similarity, estimate_tokens
from .chat_processor import load_chat, cluster_messages, get_file_excerpt
from .code_scanner import scan_codebase, find_relevant_code

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('context_bridge')

# Try to import optional dependencies
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.logging import RichHandler
    from rich import print as rich_print

    # Replace default handler with rich handler
    logging.getLogger().handlers = [RichHandler(rich_tracebacks=True)]

    USE_RICH = True
    console = Console()
except ImportError:
    USE_RICH = False
    console = None
    logger.warning("Rich library not available, using plain text output")


class ContextBridge:
    """
    Bridges chat context and code similarity analysis to create enhanced prompts.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 debug: bool = False,
                 token_limit: int = 3000,
                 similarity_threshold: float = 0.7):
        """
        Initialize the context bridge.

        Args:
            api_key: OpenAI API key for embeddings
            debug: Whether to enable debug logging
            token_limit: Maximum tokens for the generated prompt
            similarity_threshold: Threshold for similarity matching
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.debug = debug
        self.token_limit = token_limit
        self.similarity_threshold = similarity_threshold

        if debug:
            logger.setLevel(logging.DEBUG)

        # Initialize data structures
        self.chat_messages = []
        self.clustered_messages = []
        self.message_embeddings = {}

        self.code_files = {}
        self.code_embeddings = {}
        self.code_similarities = {}

        # Statistics for reporting
        self.stats = {
            'chat_messages_extracted': 0,
            'code_files_analyzed': 0,
            'relevant_code_files': 0,
            'tokens_used': 0,
        }

    def load_chat(self, chat_path: str) -> List[Dict[str, Any]]:
        """
        Load chat messages from a file.

        Args:
            chat_path: Path to the chat file

        Returns:
            List of message dictionaries
        """
        self.chat_messages = load_chat(chat_path)
        self.stats['chat_messages_extracted'] = len(self.chat_messages)
        return self.chat_messages

    def scan_codebase(self, codebase_path: str, patterns: List[str] = None) -> Dict[str, str]:
        """
        Scan a codebase directory for code files.

        Args:
            codebase_path: Path to the codebase directory
            patterns: File patterns to include (e.g., ["*.py", "*.js"])

        Returns:
            Dictionary mapping relative file paths to file content
        """
        self.code_files = scan_codebase(codebase_path, patterns)
        self.stats['code_files_analyzed'] = len(self.code_files)
        return self.code_files

    def generate_embeddings(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Generate embeddings for chat messages and code files.

        Returns:
            Tuple of (message embeddings, code embeddings)
        """
        logger.info("Generating embeddings for chat messages and code files")

        # Set up progress tracking if rich is available
        if USE_RICH:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
            ) as progress:
                # Chat messages embeddings
                if self.chat_messages:
                    chat_task = progress.add_task("Generating chat embeddings...", total=len(self.chat_messages))

                    for i, msg in enumerate(self.chat_messages):
                        content = msg.get('content', '')
                        if not content.strip():
                            continue

                        embedding = get_embedding(content, self.api_key)
                        self.message_embeddings[i] = embedding

                        progress.update(chat_task, advance=1)

                # Code files embeddings
                if self.code_files:
                    code_task = progress.add_task("Generating code embeddings...", total=len(self.code_files))

                    for path, content in self.code_files.items():
                        if not content.strip():
                            continue

                        # For longer files, use first 8000 characters to stay within limits
                        embedding = get_embedding(content[:8000], self.api_key)
                        self.code_embeddings[path] = embedding

                        progress.update(code_task, advance=1)
        else:
            # No rich progress bar
            logger.info("Generating chat message embeddings...")
            for i, msg in enumerate(self.chat_messages):
                content = msg.get('content', '')
                if not content.strip():
                    continue

                embedding = get_embedding(content, self.api_key)
                self.message_embeddings[i] = embedding

            logger.info("Generating code file embeddings...")
            for path, content in self.code_files.items():
                if not content.strip():
                    continue

                # For longer files, use first 8000 characters to stay within limits
                embedding = get_embedding(content[:8000], self.api_key)
                self.code_embeddings[path] = embedding

        logger.info(
            f"Generated embeddings for {len(self.message_embeddings)} messages and {len(self.code_embeddings)} code files")

        return self.message_embeddings, self.code_embeddings

    def find_relevant_code(self, min_similarity: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Find code files relevant to chat messages based on embedding similarity.

        Args:
            min_similarity: Minimum similarity score to consider relevant

        Returns:
            Dictionary mapping message indices to relevant code files with scores
        """
        self.code_similarities = find_relevant_code(
            self.message_embeddings, 
            self.code_embeddings, 
            min_similarity, 
            compute_similarity
        )

        self.stats['relevant_code_files'] = sum(len(files) for files in self.code_similarities.values())
        return self.code_similarities

    def cluster_messages(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Cluster and select important messages.

        Args:
            max_messages: Maximum number of messages to include

        Returns:
            List of selected messages
        """
        self.clustered_messages = cluster_messages(
            self.chat_messages, 
            self.message_embeddings, 
            self.code_similarities, 
            max_messages, 
            compute_similarity
        )

        return self.clustered_messages

    def generate_enhanced_prompt(self, max_tokens: int = 3000) -> str:
        """
        Generate an enhanced prompt with chat context and relevant code.

        Args:
            max_tokens: Maximum tokens for the prompt

        Returns:
            Enhanced prompt text
        """
        logger.info("Generating enhanced prompt")

        if not self.clustered_messages:
            logger.warning("No clustered messages. Cluster messages first.")
            return "# No messages available for prompt generation"

        sections = []
        token_count = 0

        # Add header
        header = (
            "# Enhanced Context for Conversation Continuation\n\n"
            "Below is the essential context from our previous conversation, "
            "along with relevant code files that were discussed.\n\n"
        )
        sections.append(header)
        token_count += estimate_tokens(header)

        # Process messages in order
        for i, msg in enumerate(self.clustered_messages):
            index = msg.get('index')
            role = "You" if msg.get('role') == 'assistant' else "Me"
            content = msg.get('content', '').strip()

            if not content:
                continue

            # Format the message
            if i == len(self.clustered_messages) - 1:
                formatted = f"### Most Recent Message ({role}):\n{content}\n\n"
            else:
                formatted = f"### {role}:\n{content}\n\n"

            # Check if we'd exceed token limit
            msg_tokens = estimate_tokens(formatted)
            if token_count + msg_tokens > max_tokens * 0.7:  # Reserve 30% for code
                # Add a marker
                summary = "*[Several messages omitted to save space]*\n\n"
                sections.append(summary)
                token_count += estimate_tokens(summary)

                # Skip to the last message
                if i < len(self.clustered_messages) - 1:
                    continue

            sections.append(formatted)
            token_count += msg_tokens

            # Add relevant code if available and we have tokens left
            if index in self.code_similarities and self.code_similarities[index]:
                # Get top 2 most relevant files
                top_files = sorted(
                    self.code_similarities[index].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:2]

                for file_path, similarity in top_files:
                    # Only include high-relevance files
                    if similarity < self.similarity_threshold:
                        continue

                    # Format code section
                    code_excerpt = get_file_excerpt(file_path, self.code_files.get(file_path, ""))

                    if code_excerpt:
                        code_section = (
                            f"#### Relevant Code File: `{file_path}`\n"
                            f"```{file_path.split('.')[-1]}\n"
                            f"{code_excerpt}\n"
                            f"```\n\n"
                        )

                        # Check token limit
                        code_tokens = estimate_tokens(code_section)
                        if token_count + code_tokens > max_tokens:
                            # If we can't fit the full excerpt, add a reference instead
                            ref_section = (
                                f"#### Relevant Code File: `{file_path}`\n"
                                f"*This file is relevant to the discussion but too large to include in full.*\n\n"
                            )
                            sections.append(ref_section)
                            token_count += estimate_tokens(ref_section)
                        else:
                            sections.append(code_section)
                            token_count += code_tokens

        # Add footer
        footer = (
            "---\n\n"
            "Please continue our conversation based on this enhanced context. "
            "Feel free to reference the code snippets in your response.\n"
        )
        sections.append(footer)
        token_count += estimate_tokens(footer)

        prompt = ''.join(sections)

        self.stats['tokens_used'] = token_count
        logger.info(f"Generated prompt with approximately {token_count} tokens")

        return prompt

    def print_analysis_summary(self) -> None:
        """Print a summary of the analysis to the console."""
        if USE_RICH:
            console.print("\n[bold green]=== Context Bridge Analysis Summary ===[/bold green]")
            console.print(f"📊 Analyzed [cyan]{self.stats['chat_messages_extracted']}[/] chat messages")
            console.print(f"📁 Processed [cyan]{self.stats['code_files_analyzed']}[/] code files")
            console.print(f"🔍 Found [cyan]{self.stats['relevant_code_files']}[/] relevant code references")
            console.print(f"📝 Generated prompt with [cyan]{self.stats['tokens_used']}[/] tokens")

            if self.clustered_messages:
                console.print("\n[bold yellow]Selected Messages:[/bold yellow]")
                for i, msg in enumerate(self.clustered_messages):
                    role = "User" if msg.get('role') == 'user' else "Assistant"
                    content = msg.get('content', '')
                    preview = content[:50] + ("..." if len(content) > 50 else "")
                    console.print(f"{i + 1}. [bold]{role}:[/bold] {preview}")

            if self.code_similarities:
                console.print("\n[bold yellow]Top Code References:[/bold yellow]")

                # Find the top 5 highest-similarity file references
                top_refs = []
                for msg_idx, files in self.code_similarities.items():
                    for file_path, score in files.items():
                        top_refs.append((msg_idx, file_path, score))

                top_refs.sort(key=lambda x: x[2], reverse=True)

                for msg_idx, file_path, score in top_refs[:5]:
                    msg_type = "User" if self.chat_messages[msg_idx].get('role') == 'user' else "Assistant"
                    console.print(f"[cyan]{file_path}[/] (similarity: {score:.2f}) referenced in {msg_type} message")

        else:
            print("\n=== Context Bridge Analysis Summary ===")
            print(f"- Analyzed {self.stats['chat_messages_extracted']} chat messages")
            print(f"- Processed {self.stats['code_files_analyzed']} code files")
            print(f"- Found {self.stats['relevant_code_files']} relevant code references")
            print(f"- Generated prompt with {self.stats['tokens_used']} tokens")

            if self.clustered_messages:
                print("\nSelected Messages:")
                for i, msg in enumerate(self.clustered_messages):
                    role = "User" if msg.get('role') == 'user' else "Assistant"
                    content = msg.get('content', '')
                    preview = content[:50] + ("..." if len(content) > 50 else "")
                    print(f"{i + 1}. {role}: {preview}")

            if self.code_similarities:
                print("\nTop Code References:")

                # Find the top 5 highest-similarity file references
                top_refs = []
                for msg_idx, files in self.code_similarities.items():
                    for file_path, score in files.items():
                        top_refs.append((msg_idx, file_path, score))

                top_refs.sort(key=lambda x: x[2], reverse=True)

                for msg_idx, file_path, score in top_refs[:5]:
                    msg_type = "User" if self.chat_messages[msg_idx].get('role') == 'user' else "Assistant"
                    print(f"{file_path} (similarity: {score:.2f}) referenced in {msg_type} message")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced prompts with chat context and relevant code"
    )

    parser.add_argument(
        "--chat",
        help="Path to chat export file",
        required=True
    )
    parser.add_argument(
        "--codebase",
        help="Path to codebase directory",
        required=True
    )
    parser.add_argument(
        "--output",
        default="enhanced_prompt.md",
        help="Output file for the enhanced prompt"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=3000,
        help="Maximum tokens for the prompt"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (uses OPENAI_API_KEY environment variable if not provided)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for code relevance"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the generated prompt to console"
    )

    args = parser.parse_args()

    try:
        # Initialize the context bridge
        bridge = ContextBridge(
            api_key=args.api_key,
            debug=args.debug,
            token_limit=args.tokens,
            similarity_threshold=args.threshold
        )

        # Process chat and codebase
        bridge.load_chat(args.chat)
        bridge.scan_codebase(args.codebase)

        # Generate embeddings and find relationships
        bridge.generate_embeddings()
        bridge.find_relevant_code()

        # Cluster messages and generate enhanced prompt
        bridge.cluster_messages()
        prompt = bridge.generate_enhanced_prompt(max_tokens=args.tokens)

        # Write prompt to file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(prompt)

        logger.info(f"Enhanced prompt saved to {args.output}")

        # Print analysis summary
        bridge.print_analysis_summary()

        # Print prompt if requested
        if args.print:
            if USE_RICH:
                console.print("\n[bold green]Generated Enhanced Prompt:[/bold green]\n")
                console.print(Markdown(prompt))
            else:
                print("\nGenerated Enhanced Prompt:\n")
                print(prompt)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.debug and USE_RICH:
            import traceback
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()

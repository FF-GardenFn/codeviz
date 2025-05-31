"""
chat_processor.py

Utilities for loading, parsing, and processing chat data from various formats.
"""
import json
import re
import os
import logging
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('chat_processor')

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


def load_chat(chat_path: str) -> List[Dict[str, Any]]:
    """
    Load chat messages from a file.

    Args:
        chat_path: Path to the chat file

    Returns:
        List of message dictionaries
    """
    logger.info(f"Loading chat from {chat_path}")

    try:
        # Validate the file
        if not os.path.exists(chat_path):
            raise ValueError(f"File not found: {chat_path}")

        with open(chat_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse as JSON first
        messages = []

        if content.startswith('{') or content.startswith('['):
            try:
                data = json.loads(content)

                # Different JSON formats
                if isinstance(data, list) and len(data) > 0 and 'content' in data[0]:
                    # List of message objects
                    messages = data

                elif isinstance(data, dict):
                    # ChatGPT format
                    if 'mapping' in data and 'messages' in data:
                        messages = []
                        for msg_id, msg_data in data['mapping'].items():
                            if 'message' in msg_data and msg_data['message']:
                                msg = msg_data['message']
                                if 'content' in msg and 'parts' in msg['content']:
                                    parts = msg['content']['parts']
                                    role = msg.get('author', {}).get('role', 'unknown')
                                    messages.append({
                                        'role': role,
                                        'content': ''.join(parts) if isinstance(parts, list) else str(parts)
                                    })

                    # Claude format
                    elif 'conversation_id' in data and 'messages' in data:
                        messages = [
                            {
                                'role': msg.get('type', 'unknown'),
                                'content': msg.get('text', '')
                            }
                            for msg in data['messages']
                        ]

                    # Simple format
                    elif 'messages' in data and isinstance(data['messages'], list):
                        messages = data['messages']

            except json.JSONDecodeError:
                # Not valid JSON, continue to other formats
                pass

        # If JSON parsing didn't work, try to extract from plain text
        if not messages:
            # Try to match "Human: ... Assistant: ..." pattern
            pattern = re.compile(
                r'(Human|Assistant|User|AI|Claude|ChatGPT):\s*(.*?)(?=\n(?:Human|Assistant|User|AI|Claude|ChatGPT):|$)',
                re.DOTALL)
            matches = pattern.findall(content)

            if matches:
                for role, msg_content in matches:
                    role_normalized = 'user' if role.lower() in ['human', 'user'] else 'assistant'
                    if msg_content.strip():  # Only add non-empty messages
                        messages.append({
                            'role': role_normalized,
                            'content': msg_content.strip()
                        })

            # If still no messages, treat the whole content as one message
            if not messages and content.strip():
                messages = [{'role': 'unknown', 'content': content.strip()}]

        # Filter out any empty messages
        messages = [msg for msg in messages if msg.get('content', '').strip()]

        logger.info(f"Extracted {len(messages)} messages from chat")
        return messages

    except Exception as e:
        logger.error(f"Error loading chat: {str(e)}")
        raise


def cluster_messages(messages: List[Dict[str, Any]], 
                     message_embeddings: Dict[int, List[float]], 
                     code_similarities: Dict[int, Dict[str, float]], 
                     max_messages: int = 10,
                     compute_similarity_func=None) -> List[Dict[str, Any]]:
    """
    Cluster and select important messages.

    Args:
        messages: List of chat messages
        message_embeddings: Dictionary mapping message indices to embeddings
        code_similarities: Dictionary mapping message indices to relevant code files with scores
        max_messages: Maximum number of messages to include
        compute_similarity_func: Function to compute similarity between embeddings

    Returns:
        List of selected messages
    """
    logger.info("Clustering and selecting important messages")

    if len(messages) <= max_messages:
        # If few messages, include all
        selected = []
        for i, msg in enumerate(messages):
            selected.append({
                **msg,
                'index': i,
                'importance': 1.0
            })
        return selected

    # Compute message importance
    messages_with_scores = []

    for i, msg in enumerate(messages):
        content = msg.get('content', '')

        # Length score
        length_score = min(1.0, len(content) / 2000)

        # Position score - favor first, last, and recent messages
        position = i / max(1, len(messages) - 1)
        is_first = (i == 0)
        is_last = (i == len(messages) - 1)

        if is_first or is_last:
            position_score = 1.0
        else:
            position_score = 0.3 + (0.7 * position)  # Favor recent messages

        # Role score - questions often establish context
        role_score = 1.2 if msg.get('role') == 'user' else 1.0

        # Content structure score
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        lists = len(re.findall(r'\n\s*[-*]\s', content))

        structure_score = 1.0 + (0.2 * code_blocks) + (0.05 * lists)

        # Code relevance score - if message has relevant code
        code_score = 1.0
        if i in code_similarities and code_similarities[i]:
            code_score = 1.5  # Boost messages with relevant code

        # Combined score
        importance = (
                0.2 * length_score +
                0.3 * position_score +
                0.2 * role_score +
                0.1 * structure_score +
                0.2 * code_score
        )

        messages_with_scores.append({
            **msg,
            'index': i,
            'importance': importance
        })

    # Select first and last messages
    selected_indices = {0, len(messages) - 1}

    # Greedy selection of diverse important messages
    remaining = list(range(1, len(messages) - 1))

    while len(selected_indices) < max_messages and remaining:
        best_score = -float('inf')
        best_idx = -1

        for i in remaining:
            # Base score is message importance
            score = messages_with_scores[i]['importance']

            # Reduce score if similar to already selected
            for j in selected_indices:
                if i in message_embeddings and j in message_embeddings and compute_similarity_func:
                    sim = compute_similarity_func(
                        message_embeddings[i],
                        message_embeddings[j]
                    )
                    score -= sim * 0.5  # Penalty for similarity

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected_indices.add(best_idx)
            remaining.remove(best_idx)
        else:
            break

    # Sort by original index
    selected = sorted([messages_with_scores[i] for i in selected_indices], key=lambda x: x['index'])

    logger.info(f"Selected {len(selected)} essential messages from {len(messages)} total")

    return selected


def get_file_excerpt(file_path: str, content: str, max_lines: int = 50) -> str:
    """
    Get an excerpt from a file, focusing on the most informative parts.

    Args:
        file_path: Path to the file
        content: Content of the file
        max_lines: Maximum number of lines to include

    Returns:
        File excerpt
    """
    if not content:
        return ""

    lines = content.split('\n')

    # If file is already small enough, return it all
    if len(lines) <= max_lines:
        return content

    # Find the most informative sections (imports, class/function definitions, etc.)
    # This is a simplified approach - a more sophisticated method could be used

    # Score each line based on informativeness
    line_scores = []
    for i, line in enumerate(lines):
        score = 0

        # Imports are very informative
        if re.match(r'^\s*import\s+|^\s*from\s+.*\s+import\s+', line):
            score += 10

        # Class and function definitions are informative
        if re.match(r'^\s*(?:class|def)\s+\w+', line):
            score += 8

        # Variable assignments can be informative
        if re.match(r'^\s*\w+\s*=', line):
            score += 3

        # Comments might be informative
        if re.match(r'^\s*#', line):
            score += 2

        # Beginning and end of file often have important context
        if i < 10 or i >= len(lines) - 10:
            score += 5

        line_scores.append((i, score, line))

    # Sort by score, highest first
    line_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top-scored lines, up to max_lines
    top_lines = line_scores[:max_lines]

    # Sort by original line number to maintain order
    top_lines.sort(key=lambda x: x[0])

    # Build the excerpt
    excerpt_lines = []
    prev_line_num = -2

    for line_num, _, line in top_lines:
        # Add a marker if lines were skipped
        if line_num > prev_line_num + 1:
            excerpt_lines.append(f"... [lines {prev_line_num + 2}-{line_num} omitted] ...")

        excerpt_lines.append(line)
        prev_line_num = line_num

    return '\n'.join(excerpt_lines)
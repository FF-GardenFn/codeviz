"""
code_scanner.py

Utilities for scanning and processing code files from a codebase.
"""
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('code_scanner')

# Try to import optional dependencies
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    USE_RICH = True
    console = Console()
except ImportError:
    USE_RICH = False
    console = None
    logger.warning("Rich library not available, using plain text output")


def scan_codebase(codebase_path: str, patterns: List[str] = None) -> Dict[str, str]:
    """
    Scan a codebase directory for code files.

    Args:
        codebase_path: Path to the codebase directory
        patterns: File patterns to include (e.g., ["*.py", "*.js"])

    Returns:
        Dictionary mapping relative file paths to file content
    """
    logger.info(f"Scanning codebase at {codebase_path}")

    if not os.path.exists(codebase_path):
        raise ValueError(f"Codebase directory not found: {codebase_path}")

    # Default patterns if none provided
    if not patterns:
        patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.html", "*.css", "*.json", "*.md"]

    code_files = {}
    base_path = Path(codebase_path)

    # Set up progress tracking if rich is available
    if USE_RICH:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)

            for pattern in patterns:
                for file_path in base_path.glob(f"**/{pattern}"):
                    # Skip hidden files and directories
                    if any(part.startswith('.') for part in file_path.parts):
                        continue

                    relative_path = file_path.relative_to(base_path)

                    try:
                        content = file_path.read_text(encoding='utf-8')
                        code_files[str(relative_path)] = content
                        progress.update(task, description=f"Found {len(code_files)} files...")
                    except Exception as e:
                        logger.warning(f"Error reading {relative_path}: {str(e)}")

            progress.update(task, completed=True, description=f"Found {len(code_files)} files")
    else:
        # No rich progress bar
        for pattern in patterns:
            for file_path in base_path.glob(f"**/{pattern}"):
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue

                relative_path = file_path.relative_to(base_path)

                try:
                    content = file_path.read_text(encoding='utf-8')
                    code_files[str(relative_path)] = content
                except Exception as e:
                    logger.warning(f"Error reading {relative_path}: {str(e)}")

    logger.info(f"Found {len(code_files)} code files")
    return code_files


def find_relevant_code(message_embeddings: Dict[int, List[float]], 
                       code_embeddings: Dict[str, List[float]], 
                       min_similarity: float = 0.5,
                       compute_similarity_func=None) -> Dict[int, Dict[str, float]]:
    """
    Find code files relevant to chat messages based on embedding similarity.

    Args:
        message_embeddings: Dictionary mapping message indices to embeddings
        code_embeddings: Dictionary mapping file paths to embeddings
        min_similarity: Minimum similarity score to consider relevant
        compute_similarity_func: Function to compute similarity between embeddings

    Returns:
        Dictionary mapping message indices to relevant code files with scores
    """
    logger.info("Finding code files relevant to chat messages")

    if not message_embeddings or not code_embeddings:
        logger.warning("No embeddings available. Generate embeddings first.")
        return {}

    if not compute_similarity_func:
        logger.warning("No similarity function provided. Cannot compute similarities.")
        return {}

    relevant_code = {}

    # For each message, find relevant code files
    for msg_idx, msg_embedding in message_embeddings.items():
        relevant_files = {}

        for file_path, file_embedding in code_embeddings.items():
            similarity = compute_similarity_func(msg_embedding, file_embedding)

            if similarity >= min_similarity:
                relevant_files[file_path] = similarity

        # Sort by similarity (highest first) and keep top results
        sorted_files = {k: v for k, v in sorted(
            relevant_files.items(),
            key=lambda item: item[1],
            reverse=True
        )}

        if sorted_files:
            relevant_code[msg_idx] = sorted_files

    logger.info(f"Found {sum(len(files) for files in relevant_code.values())} relevant code file references")
    return relevant_code
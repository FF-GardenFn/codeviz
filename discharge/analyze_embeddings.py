"""
analyze_embeddings.py

Analyze semantic similarity between files based on their embeddings.
Finds similar files, identifies clusters, and provides refactoring suggestions.

Usage:
  python analyze_embeddings.py \
      --report embeddings.json \
      --top_k 5 \
      --output similarity.json

The script is flexible and can handle different embedding file formats:
- CodeViz report format
- Direct path-to-embedding mapping
- List of objects with path and embedding fields
"""
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os

try:
    import numpy as np
    from numpy.typing import NDArray

    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    import math

try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rich_print

    USE_RICH = True
except ImportError:
    USE_RICH = False


def extract_embeddings_from_report(report_path: str) -> Tuple[List[str], Any]:
    """
    Extract embeddings from a CodeViz report file.

    Args:
        report_path: Path to the CodeViz report JSON file

    Returns:
        Tuple of (file paths, embedding vectors)
    """
    console = Console() if USE_RICH else None

    if console:
        console.print(f"📊 Loading file from [cyan]{report_path}[/]...")
    else:
        print(f"📊 Loading file from {report_path}...")

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            try:
                report = json.load(f)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues by reading up to the error position
                f.seek(0)
                partial_content = f.read(e.pos)
                try:
                    # Try to parse the content up to the error
                    report = json.loads(partial_content)
                    if console:
                        console.print(f"[yellow]Warning: Fixed partial JSON parsing up to position {e.pos}[/]")
                    else:
                        print(f"Warning: Fixed partial JSON parsing up to position {e.pos}")
                except:
                    # If that doesn't work, try one more approach
                    try:
                        # Try to find the last complete object
                        last_brace = partial_content.rstrip().rfind('}')
                        if last_brace > 0:
                            clean_content = partial_content[:last_brace + 1]
                            report = json.loads(clean_content)
                            if console:
                                console.print(f"[yellow]Warning: Parsed JSON up to last complete object[/]")
                            else:
                                print(f"Warning: Parsed JSON up to last complete object")
                        else:
                            raise
                    except:
                        sys.exit(f"Error loading file: {e}. The file appears to contain invalid JSON.")
    except Exception as e:
        sys.exit(f"Error loading file: {e}")

    # Print the root keys to help with debugging
    if console:
        console.print(
            f"File contains these root keys: [cyan]{', '.join(report.keys() if isinstance(report, dict) else ['<not a dictionary>'])}[/]")
    else:
        if isinstance(report, dict):
            print(f"File contains these root keys: {', '.join(report.keys())}")
        else:
            print(f"File is not a dictionary structure, but a {type(report).__name__}")

    # Try different known formats
    paths = []
    embeddings = []

    # Format 1: Standard CodeViz report with summaries
    if isinstance(report, dict) and 'summaries' in report:
        summaries = report['summaries']
        for summary in summaries:
            if 'embedding' in summary and summary['embedding']:
                paths.append(summary.get('path', f"file_{len(paths)}"))
                embeddings.append(summary['embedding'])

    # Format 2: Direct mapping of path to embedding
    elif isinstance(report, dict) and len(report) > 0:
        # Check if values look like embeddings (lists of numbers)
        first_key = next(iter(report))
        if isinstance(report[first_key], list) and all(isinstance(x, (int, float)) for x in report[first_key][:5]):
            for path, embedding in report.items():
                paths.append(path)
                embeddings.append(embedding)

    # Format 3: List of objects with path and embedding
    elif isinstance(report, list) and len(report) > 0:
        if all(isinstance(item, dict) for item in report):
            # Check for common embedding key patterns
            embedding_keys = ['embedding', 'embeddings', 'vector', 'vectors']
            path_keys = ['path', 'file', 'filename', 'name', 'id']

            # Find which keys are present
            sample = report[0]
            emb_key = next((k for k in embedding_keys if k in sample), None)
            path_key = next((k for k in path_keys if k in sample), None)

            if emb_key and path_key:
                for item in report:
                    if item.get(emb_key):
                        paths.append(item.get(path_key, f"file_{len(paths)}"))
                        embeddings.append(item[emb_key])

    embed_count = len(embeddings)

    if embed_count == 0:
        if console:
            console.print("[bold red]No embeddings found in the file.[/]")
            console.print("The file may not contain embeddings or has an unexpected format.")
            console.print("Try inspecting the file structure to identify the correct format.")
        else:
            print("No embeddings found in the file.")
            print("The file may not contain embeddings or has an unexpected format.")
            print("Try inspecting the file structure to identify the correct format.")
        sys.exit(1)

    if console:
        console.print(f"✅ Found embeddings for [green]{embed_count}[/] files")
    else:
        print(f"✅ Found embeddings for {embed_count} files")

    if USE_NUMPY:
        try:
            vecs = np.array(embeddings)
        except Exception as e:
            sys.exit(f"Failed to build embedding matrix: {e}")
        return paths, vecs
    else:
        return paths, embeddings


def load_embedding_file(path: str) -> Tuple[List[str], Any]:
    """
    Load embeddings from a dedicated embeddings JSON file (path -> vector mapping).

    Args:
        path: Path to the embeddings JSON file

    Returns:
        Tuple of (file paths, embedding vectors)
    """
    console = Console() if USE_RICH else None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        sys.exit(f"Error loading embeddings file: {e}")

    if not data:
        sys.exit("Empty embeddings file")

    # Check format - either dict or list
    if isinstance(data, dict):
        # Format: {"path": [vector]}
        paths = list(data.keys())
        vecs_list = [data[p] for p in paths]
    elif isinstance(data, list) and 'path' in data[0] and 'embedding' in data[0]:
        # Format: [{"path": "file.py", "embedding": [vector]}]
        paths = [item['path'] for item in data]
        vecs_list = [item['embedding'] for item in data]
    else:
        sys.exit("Unrecognized embeddings format")

    if console:
        console.print(f"✅ Loaded embeddings for [green]{len(paths)}[/] files")
    else:
        print(f"✅ Loaded embeddings for {len(paths)} files")

    if USE_NUMPY:
        try:
            vecs = np.array(vecs_list)
        except Exception as e:
            sys.exit(f"Failed to build embedding matrix: {e}")
        return paths, vecs
    else:
        return paths, vecs_list


def compute_cosine_similarity(vecs) -> Any:
    """
    Compute cosine similarity matrix from embedding vectors.

    Args:
        vecs: Embedding vectors (numpy array or list of lists)

    Returns:
        Similarity matrix
    """
    console = Console() if USE_RICH else None

    if console:
        console.print("🧮 Computing cosine similarity matrix...")
    else:
        print("🧮 Computing cosine similarity matrix...")

    if USE_NUMPY:
        # Normalize vectors
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normed = vecs / norms
        # Compute similarity matrix
        return normed @ normed.T  # Matrix multiplication
    else:
        # Fallback pure Python implementation
        n = len(vecs)
        # Compute norms
        norms = []
        for v in vecs:
            norm = math.sqrt(sum(x * x for x in v))
            norms.append(norm if norm != 0 else 1.0)
        # Compute similarity matrix
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dot = sum(a * b for a, b in zip(vecs[i], vecs[j]))
                sim[i][j] = dot / (norms[i] * norms[j])
        return sim


def top_k_neighbors(sim_matrix, paths: List[str], k: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each file, find top-k semantically similar files (excluding self).

    Args:
        sim_matrix: Similarity matrix
        paths: List of file paths
        k: Number of neighbors to find

    Returns:
        Dictionary mapping file paths to their neighbors with similarity scores
    """
    console = Console() if USE_RICH else None

    if console:
        console.print(f"🔍 Finding top-{k} similar files for each file...")
    else:
        print(f"🔍 Finding top-{k} similar files for each file...")

    results = {}
    n = len(paths)

    for i, path in enumerate(paths):
        if USE_NUMPY:
            sims = sim_matrix[i, :].copy()
            # Exclude self
            sims[i] = -float('inf')
            # Get top-k indices
            idxs = np.argsort(sims)[-k:][::-1]
        else:
            sims = list(sim_matrix[i])
            # Exclude self
            sims[i] = -float('inf')
            # Get top-k indices
            idxs = sorted(range(n), key=lambda j: sims[j], reverse=True)[:k]

        neighbors = []
        for j in idxs:
            score = float(sim_matrix[i, j] if USE_NUMPY else sim_matrix[i][j])
            if score > 0.2:  # Only include meaningful similarities
                neighbors.append({
                    "path": paths[j],
                    "score": score,
                    "language": os.path.splitext(paths[j])[1].lstrip('.') if '.' in paths[j] else "unknown"
                })

        if neighbors:  # Only include files with at least one neighbor
            results[path] = neighbors

    return results


def identify_clusters(similarity_results: Dict[str, List[Dict[str, Any]]], threshold: float = 0.7) -> List[List[str]]:
    """
    Identify clusters of highly similar files based on similarity results.

    Args:
        similarity_results: Dictionary mapping files to similar files
        threshold: Similarity threshold for clustering

    Returns:
        List of file clusters
    """
    # Build an adjacency list where an edge exists if similarity > threshold
    adjacency = {}
    for file, neighbors in similarity_results.items():
        adjacency[file] = []
        for neighbor in neighbors:
            if neighbor["score"] >= threshold:
                adjacency[file].append(neighbor["path"])

    # Use a simple connected components algorithm to find clusters
    clusters = []
    visited = set()

    for file in adjacency:
        if file in visited:
            continue

        # Start a new cluster with DFS
        cluster = []
        stack = [file]
        while stack:
            current = stack.pop()
            if current in visited:
                continue

            visited.add(current)
            cluster.append(current)

            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(cluster) > 1:  # Only include clusters with multiple files
            clusters.append(cluster)

    # Sort clusters by size (largest first)
    return sorted(clusters, key=len, reverse=True)


def get_refactoring_suggestions(similarity_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Generate refactoring suggestions based on similarity patterns.

    Args:
        similarity_results: Dictionary mapping files to similar files

    Returns:
        List of refactoring suggestions
    """
    suggestions = []

    # Files with very high similarity to multiple others - potential duplication
    for file, neighbors in similarity_results.items():
        high_sim_neighbors = [n for n in neighbors if n["score"] > 0.85]
        if len(high_sim_neighbors) >= 2:
            suggestions.append({
                "type": "potential_duplication",
                "file": file,
                "similar_files": [n["path"] for n in high_sim_neighbors],
                "message": f"High similarity with multiple files suggests potential code duplication"
            })

    # Cross-language similarity - potential reimplementation
    for file, neighbors in similarity_results.items():
        file_ext = os.path.splitext(file)[1].lstrip('.') if '.' in file else "unknown"
        cross_lang_neighbors = [
            n for n in neighbors
            if n["language"] != file_ext and n["score"] > 0.7
        ]

        if cross_lang_neighbors:
            suggestions.append({
                "type": "cross_language_similarity",
                "file": file,
                "similar_files": [n["path"] for n in cross_lang_neighbors],
                "message": f"High similarity with files in different languages suggests functionality reimplementation"
            })

    return suggestions


def print_summary(results: Dict[str, List[Dict[str, Any]]], clusters: List[List[str]],
                  suggestions: List[Dict[str, Any]]):
    """
    Print a summary of analysis results to the console.

    Args:
        results: Similarity results
        clusters: Identified file clusters
        suggestions: Refactoring suggestions
    """
    if not USE_RICH:
        print("\n=== Analysis Summary ===")
        print(f"- Analyzed {len(results)} files")
        print(f"- Found {len(clusters)} clusters of similar files")
        print(f"- Generated {len(suggestions)} refactoring suggestions")

        if clusters:
            print("\nTop Clusters:")
            for i, cluster in enumerate(clusters[:3], 1):
                print(f"{i}. Cluster with {len(cluster)} files:")
                for file in cluster[:3]:
                    print(f"   - {file}")
                if len(cluster) > 3:
                    print(f"   - ...and {len(cluster) - 3} more files")

        if suggestions:
            print("\nTop Refactoring Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"{i}. {suggestion['message']}")
                print(f"   File: {suggestion['file']}")
                print(f"   Similar to: {', '.join(suggestion['similar_files'][:2])}")
                if len(suggestion['similar_files']) > 2:
                    print(f"   ...and {len(suggestion['similar_files']) - 2} more files")
        return

    # Rich formatting
    console = Console()

    console.print("\n[bold green]=== CodeViz Similarity Analysis Summary ===[/bold green]")
    console.print(f"📊 Analyzed [cyan]{len(results)}[/] files")
    console.print(f"🔍 Found [cyan]{len(clusters)}[/] clusters of similar files")
    console.print(f"💡 Generated [cyan]{len(suggestions)}[/] refactoring suggestions")

    if clusters:
        console.print("\n[bold yellow]Top File Clusters:[/bold yellow]")
        for i, cluster in enumerate(clusters[:3], 1):
            console.print(f"{i}. [bold]Cluster with {len(cluster)} files:[/bold]")
            for file in cluster[:3]:
                console.print(f"   [cyan]→[/] {file}")
            if len(cluster) > 3:
                console.print(f"   [dim]...and {len(cluster) - 3} more files[/dim]")

    if suggestions:
        console.print("\n[bold yellow]Refactoring Suggestions:[/bold yellow]")

        table = Table(show_header=True)
        table.add_column("Type", style="bold")
        table.add_column("File", style="cyan")
        table.add_column("Similar Files", style="green")
        table.add_column("Suggestion", style="yellow")

        for suggestion in suggestions[:5]:
            similar_files = ", ".join(suggestion['similar_files'][:2])
            if len(suggestion['similar_files']) > 2:
                similar_files += f" (+{len(suggestion['similar_files']) - 2} more)"

            table.add_row(
                suggestion['type'].replace('_', ' ').title(),
                suggestion['file'],
                similar_files,
                suggestion['message']
            )

        console.print(table)


def generate_prompt_from_similarity_report(results: Dict[str, List[Dict[str, Any]]], 
                                          clusters: List[List[str]],
                                          suggestions: List[Dict[str, Any]],
                                          max_files_per_cluster: int = 10,
                                          max_similar_files: int = 5) -> str:
    """
    Generate a markdown prompt from similarity analysis results.

    Args:
        results: Similarity results
        clusters: Identified file clusters
        suggestions: Refactoring suggestions
        max_files_per_cluster: Maximum number of files to show per cluster
        max_similar_files: Maximum number of similar files to show per file

    Returns:
        Markdown prompt text
    """
    sections = []

    # Add header
    header = (
        "# Code Similarity Analysis Report\n\n"
        "This report provides an analysis of semantic similarities between files in your codebase. "
        "It identifies clusters of related files and potential opportunities for refactoring.\n\n"
    )
    sections.append(header)

    # Add summary section
    summary = (
        "## Summary\n\n"
        f"- **Files Analyzed**: {len(results)}\n"
        f"- **Clusters Identified**: {len(clusters)}\n"
        f"- **Refactoring Suggestions**: {len(suggestions)}\n\n"
    )
    sections.append(summary)

    # Add clusters section
    if clusters:
        clusters_section = "## File Clusters\n\n"
        clusters_section += "These clusters represent groups of files that are semantically similar and likely related in functionality.\n\n"

        for i, cluster in enumerate(clusters, 1):
            clusters_section += f"### Cluster {i}: {len(cluster)} files\n\n"

            # Show files in the cluster
            for j, file in enumerate(cluster[:max_files_per_cluster]):
                clusters_section += f"- `{file}`\n"

            if len(cluster) > max_files_per_cluster:
                clusters_section += f"- *...and {len(cluster) - max_files_per_cluster} more files*\n"

            clusters_section += "\n"

        sections.append(clusters_section)

    # Add file similarities section
    similarities_section = "## File Similarities\n\n"
    similarities_section += "This section shows the most similar files for each file in the codebase.\n\n"

    # Sort files by name for consistent output
    sorted_files = sorted(results.keys())

    for file in sorted_files[:20]:  # Limit to 20 files to keep the prompt manageable
        similarities_section += f"### `{file}`\n\n"

        similar_files = results[file][:max_similar_files]
        if not similar_files:
            similarities_section += "*No similar files found.*\n\n"
            continue

        similarities_section += "| Similar File | Similarity Score |\n"
        similarities_section += "|-------------|------------------|\n"

        for similar in similar_files:
            similarities_section += f"| `{similar['path']}` | {similar['score']:.2f} |\n"

        similarities_section += "\n"

    if len(sorted_files) > 20:
        similarities_section += f"*...and {len(sorted_files) - 20} more files*\n\n"

    sections.append(similarities_section)

    # Add refactoring suggestions section
    if suggestions:
        suggestions_section = "## Refactoring Suggestions\n\n"
        suggestions_section += "Based on the similarity analysis, here are some potential opportunities for refactoring:\n\n"

        for i, suggestion in enumerate(suggestions, 1):
            suggestions_section += f"### Suggestion {i}: {suggestion['type'].replace('_', ' ').title()}\n\n"
            suggestions_section += f"**File**: `{suggestion['file']}`\n\n"
            suggestions_section += f"**Similar Files**:\n"

            for similar in suggestion['similar_files'][:5]:
                suggestions_section += f"- `{similar}`\n"

            if len(suggestion['similar_files']) > 5:
                suggestions_section += f"- *...and {len(suggestion['similar_files']) - 5} more files*\n"

            suggestions_section += f"\n**Suggestion**: {suggestion['message']}\n\n"

        sections.append(suggestions_section)

    # Add guidance section
    guidance = (
        "## Analysis Guidance\n\n"
        "### How to Use This Report\n\n"
        "1. **Examine the clusters** to understand which files are semantically related\n"
        "2. **Review the file similarities** to see which files are most similar to each other\n"
        "3. **Consider the refactoring suggestions** to identify potential improvements\n\n"

        "### Questions to Consider\n\n"
        "1. Are there files with similar functionality that could be consolidated?\n"
        "2. Are there opportunities to extract common code into shared utilities?\n"
        "3. Does the clustering align with your understanding of the codebase structure?\n"
        "4. Are there unexpected similarities that suggest hidden dependencies?\n\n"
    )
    sections.append(guidance)

    # Add footer
    footer = (
        "---\n\n"
        "Generated by CodeViz Similarity Analysis\n"
    )
    sections.append(footer)

    return ''.join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze file similarities from embeddings"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-r", "--report",
        help="Path to JSON file with embeddings"
    )
    input_group.add_argument(
        "-e", "--embeddings",
        help="Path to embeddings JSON file (legacy option, same as --report)"
    )

    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=5,
        help="Number of top similar neighbors to report per file (default: 5)"
    )
    parser.add_argument(
        "-o", "--output",
        default="similarity-report.json",
        help="Output JSON file for similarity results (default: similarity-report.json)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for clustering (default: 0.7)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Print debug information about the input file"
    )

    args = parser.parse_args()

    # Load embeddings from either source (they're now functionally the same)
    if args.report:
        paths, vecs = extract_embeddings_from_report(args.report)
    else:
        paths, vecs = extract_embeddings_from_report(args.embeddings)

    # Debug info
    if args.debug:
        console = Console() if USE_RICH else None
        if console:
            console.print(f"\n[bold]Debug Information:[/bold]")
            console.print(f"Number of files: [cyan]{len(paths)}[/]")
            console.print(f"Embedding dimensions: [cyan]{vecs.shape[1] if USE_NUMPY else len(vecs[0])}[/]")
            console.print(f"Sample paths: [cyan]{paths[:3]}[/]")
        else:
            print("\nDebug Information:")
            print(f"Number of files: {len(paths)}")
            print(f"Embedding dimensions: {vecs.shape[1] if USE_NUMPY else len(vecs[0])}")
            print(f"Sample paths: {paths[:3]}")

    # Compute similarity matrix
    sim_matrix = compute_cosine_similarity(vecs)

    # Find top-k neighbors for each file
    results = top_k_neighbors(sim_matrix, paths, args.top_k)

    # Identify clusters and generate suggestions
    clusters = identify_clusters(results, args.threshold)
    suggestions = get_refactoring_suggestions(results)

    # Build enriched output
    output = {
        "file_similarities": results,
        "clusters": clusters,
        "refactoring_suggestions": suggestions,
        "metadata": {
            "files_analyzed": len(paths),
            "similarity_threshold": args.threshold,
            "top_k": args.top_k
        }
    }

    # Write results to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # Print summary to console
    if USE_RICH:
        rich_print(f"✅ Wrote similarity analysis to [cyan]{args.output}[/]")
    else:
        print(f"✅ Wrote similarity analysis to {args.output}")

    print_summary(results, clusters, suggestions)


if __name__ == '__main__':
    main()

from pathlib import Path
import json
import sys
from typing import Optional, List
import typer
import structlog

from .analyzers.project_analyzer import ProjectAnalyzer
from .services.directory_tree import DirectoryTree
from .config import settings
from .logging_conf import configure_logging
from .discharge.context_bridge import ContextBridge
from .discharge.analyze_embeddings import (
    extract_embeddings_from_report, 
    compute_cosine_similarity,
    top_k_neighbors,
    identify_clusters,
    get_refactoring_suggestions,
    print_summary,
    generate_prompt_from_similarity_report
)

# Initialize app
app = typer.Typer(
    add_completion=True,
    help="📊 CodeViz – Comprehensive Code Analysis and Context Bridging Tool"
)

logger = configure_logging()


@app.command()
def analyse(
        root: Path = typer.Argument(
            ".",
            exists=True,
            dir_okay=True,
            help="Project root directory to analyze"
        ),
        out: Path = typer.Option(
            "codeviz-report.json",
            "--out", "-o",
            help="Path to write JSON report"
        ),
        tree: bool = typer.Option(
            False,
            "--tree", "-t",
            help="Print directory tree"
        ),
        summary: bool = typer.Option(
            False,
            "--summary", "-s",
            help="Generate per-file summaries"
        ),
        context: bool = typer.Option(
            False,
            "--context", "-c",
            help="Generate project context summary"
        ),
        embeddings: bool = typer.Option(
            False,
            "--embeddings", "-e",
            help="Generate OpenAI embeddings"
        ),
        api_key: Optional[str] = typer.Option(
            None,
            "--api-key",
            help="OpenAI API key (overrides environment variable)"
        ),
):
    """
    Analyze a project directory and generate a comprehensive report.

    This command will scan the project structure, analyze dependencies between
    files, and optionally generate summaries and embeddings.
    """
    logger.info(f"Starting analysis of {root.resolve()}")

    # Initialize the project analyzer
    proj = ProjectAnalyzer(root)

    # Walk the project directory to find dependencies
    typer.echo("📂 Analyzing project structure...")
    proj.walk()

    # Generate file summaries if requested
    if summary or embeddings:  # Embeddings requires summaries
        typer.echo("📝 Generating file summaries...")
        proj.generate_file_summaries()

    # Generate embeddings if requested
    if embeddings:
        # Use API key from command line or settings
        openai_key = api_key or settings.openai_api_key

        if not openai_key:
            typer.echo("⚠️ No OpenAI API key configured – skipping embeddings", err=True)
            typer.echo("   Set CODEVIZ_OPENAI_API_KEY environment variable or use --api-key option")
        else:
            typer.echo("🧠 Generating OpenAI embeddings...")
            proj.generate_openai_embeddings(openai_key)

    # Build and write the report
    report = proj.build_report()
    out.write_text(json.dumps(report, indent=2))
    typer.echo(f"✅ Report written to {out.resolve()}")
    # Dump raw embeddings to a separate file if requested
    if embeddings:
        raw_embeddings: dict[str, list[float]] = {}
        for summary in proj.summaries.values():
            if summary.embedding is not None:
                raw_embeddings[str(summary.path)] = summary.embedding  # type: ignore
        emb_file = out.with_name(f"{out.stem}-embeddings.json")
        emb_file.write_text(json.dumps(raw_embeddings, indent=2))
        typer.echo(f"✅ Embeddings written to {emb_file.resolve()}")

    # Print directory tree if requested
    if tree:
        typer.echo("\n📁 Directory structure:\n")
        typer.echo(report.get("directory_tree", ""))

    # Print context summary if requested
    if context and "context_summary" in report:
        typer.echo("\n🔍 Context Summary:\n")

        cs = report["context_summary"]

        typer.echo("Top connected files:")
        for file in cs.get("most_connected_files", [])[:5]:
            typer.echo(f"  • {file['file']} ({file['connections']} connections)")

        typer.echo("\nTop imported modules:")
        for module in cs.get("top_imported_modules", [])[:5]:
            typer.echo(f"  • {module['module']} (imported {module['imported_by_count']} times)")

        typer.echo("\nImport chains (examples):")
        for chain in cs.get("import_chains", [])[:3]:
            typer.echo(f"  • {chain}")

        typer.echo(f"\nTotal files analyzed: {cs.get('total_files_analyzed', 0)}")
        typer.echo(f"Total connections: {cs.get('total_connections', 0)}")


@app.command()
def tree(
        path: Path = typer.Argument(
            ".",
            exists=True,
            dir_okay=True,
            help="Directory to display tree for"
        ),
):
    """
    Generate and display a directory tree.
    """
    tree = DirectoryTree(path)
    typer.echo(tree.generate_tree())


@app.command()
def bridge(
        chat: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            help="Path to chat export file"
        ),
        codebase: Path = typer.Argument(
            ".",
            exists=True,
            dir_okay=True,
            help="Path to codebase directory"
        ),
        output: Path = typer.Option(
            "enhanced_prompt.md",
            "--output", "-o",
            help="Output file for the enhanced prompt"
        ),
        tokens: int = typer.Option(
            3000,
            "--tokens", "-t",
            help="Maximum tokens for the prompt"
        ),
        api_key: Optional[str] = typer.Option(
            None,
            "--api-key",
            help="OpenAI API key (uses OPENAI_API_KEY environment variable if not provided)"
        ),
        threshold: float = typer.Option(
            0.7,
            "--threshold",
            help="Similarity threshold for code relevance"
        ),
        debug: bool = typer.Option(
            False,
            "--debug", "-d",
            help="Enable debug logging"
        ),
        print_prompt: bool = typer.Option(
            False,
            "--print", "-p",
            help="Print the generated prompt to console"
        ),
):
    """
    Generate enhanced prompts with chat context and relevant code.

    This command bridges code similarity analysis with chat context distillation
    to create rich, contextually-aware prompts that incorporate both conversation
    history and relevant code.
    """
    typer.echo(f"🔍 Analyzing chat from {chat} and codebase at {codebase}")

    try:
        # Initialize the context bridge
        bridge = ContextBridge(
            api_key=api_key,
            debug=debug,
            token_limit=tokens,
            similarity_threshold=threshold
        )

        # Process chat and codebase
        typer.echo("📝 Loading chat messages...")
        bridge.load_chat(str(chat))

        typer.echo("📂 Scanning codebase...")
        bridge.scan_codebase(str(codebase))

        # Generate embeddings and find relationships
        typer.echo("🧠 Generating embeddings...")
        bridge.generate_embeddings()

        typer.echo("🔗 Finding relevant code...")
        bridge.find_relevant_code()

        # Cluster messages and generate enhanced prompt
        typer.echo("📊 Clustering messages...")
        bridge.cluster_messages()

        typer.echo("✨ Generating enhanced prompt...")
        prompt = bridge.generate_enhanced_prompt(max_tokens=tokens)

        # Write prompt to file
        output.write_text(prompt, encoding='utf-8')
        typer.echo(f"✅ Enhanced prompt saved to {output.resolve()}")

        # Print analysis summary
        bridge.print_analysis_summary()

        # Print prompt if requested
        if print_prompt:
            typer.echo("\nGenerated Enhanced Prompt:\n")
            typer.echo(prompt)

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@app.command()
def analyze_embeddings(
        report: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            help="Path to JSON file with embeddings"
        ),
        output: Path = typer.Option(
            "similarity-report.json",
            "--output", "-o",
            help="Output JSON file for similarity results"
        ),
        top_k: int = typer.Option(
            5,
            "--top-k", "-k",
            help="Number of top similar neighbors to report per file"
        ),
        threshold: float = typer.Option(
            0.7,
            "--threshold", "-t",
            help="Similarity threshold for clustering"
        ),
        debug: bool = typer.Option(
            False,
            "--debug", "-d",
            help="Print debug information about the input file"
        ),
):
    """
    Analyze semantic similarity between files based on their embeddings.

    This command finds similar files, identifies clusters, and provides
    refactoring suggestions based on embedding similarity.
    """
    typer.echo(f"📊 Analyzing embeddings from {report}")

    try:
        # Load embeddings
        paths, vecs = extract_embeddings_from_report(str(report))

        # Debug info
        if debug:
            typer.echo(f"\nDebug Information:")
            typer.echo(f"Number of files: {len(paths)}")
            typer.echo(f"Embedding dimensions: {len(vecs[0]) if vecs else 0}")
            typer.echo(f"Sample paths: {paths[:3]}")

        # Compute similarity matrix
        typer.echo("🧮 Computing similarity matrix...")
        sim_matrix = compute_cosine_similarity(vecs)

        # Find top-k neighbors for each file
        typer.echo(f"🔍 Finding top-{top_k} similar files...")
        results = top_k_neighbors(sim_matrix, paths, top_k)

        # Identify clusters and generate suggestions
        typer.echo("🔍 Identifying clusters and generating suggestions...")
        clusters = identify_clusters(results, threshold)
        suggestions = get_refactoring_suggestions(results)

        # Build enriched output
        output_data = {
            "file_similarities": results,
            "clusters": clusters,
            "refactoring_suggestions": suggestions,
            "metadata": {
                "files_analyzed": len(paths),
                "similarity_threshold": threshold,
                "top_k": top_k
            }
        }

        # Write results to file
        output.write_text(json.dumps(output_data, indent=2), encoding='utf-8')
        typer.echo(f"✅ Similarity analysis written to {output.resolve()}")

        # Print summary
        print_summary(results, clusters, suggestions)

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@app.command()
def similarity_to_prompt(
        report: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            help="Path to similarity report JSON file"
        ),
        output: Path = typer.Option(
            "similarity-prompt.md",
            "--output", "-o",
            help="Output file for the generated prompt"
        ),
        max_files_per_cluster: int = typer.Option(
            10,
            "--max-files-cluster", "-m",
            help="Maximum number of files to show per cluster"
        ),
        max_similar_files: int = typer.Option(
            5,
            "--max-similar", "-s",
            help="Maximum number of similar files to show per file"
        ),
        print_prompt: bool = typer.Option(
            False,
            "--print", "-p",
            help="Print the generated prompt to console"
        ),
):
    """
    Convert a similarity report to a markdown prompt.

    This command takes a similarity report generated by the analyze-embeddings
    command and converts it to a markdown prompt that can be used with an LLM.
    The prompt includes information about file similarities, clusters, and
    refactoring suggestions.
    """
    typer.echo(f"📊 Converting similarity report from {report} to prompt")

    try:
        # Load the similarity report
        with open(report, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the components
        results = data.get("file_similarities", {})
        clusters = data.get("clusters", [])
        suggestions = data.get("refactoring_suggestions", [])

        # Generate the prompt
        typer.echo("✨ Generating prompt...")
        prompt = generate_prompt_from_similarity_report(
            results, 
            clusters, 
            suggestions,
            max_files_per_cluster,
            max_similar_files
        )

        # Write the prompt to file
        output.write_text(prompt, encoding='utf-8')
        typer.echo(f"✅ Prompt saved to {output.resolve()}")

        # Print the prompt if requested
        if print_prompt:
            typer.echo("\nGenerated Prompt:\n")
            typer.echo(prompt)

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()

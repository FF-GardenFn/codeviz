"""
report_to_prompt.py

A script to convert a CodeViz analysis report into a ready-to-use prompt for LLMs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_report(report_path: str) -> Dict[str, Any]:
    """
    Load a CodeViz analysis report from a JSON file.
    
    Args:
        report_path: Path to the report file
        
    Returns:
        The report as a dictionary
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}", file=sys.stderr)
        sys.exit(1)

def generate_project_overview(report: Dict[str, Any]) -> str:
    """
    Generate a project overview section based on the report.
    
    Args:
        report: The CodeViz analysis report
        
    Returns:
        A string containing the project overview
    """
    root = report.get("root", "Unknown")
    
    # Get context summary if available
    context_summary = report.get("context_summary", {})
    most_connected_files = context_summary.get("most_connected_files", [])
    top_imported_modules = context_summary.get("top_imported_modules", [])
    total_files = context_summary.get("total_files_analyzed", 0)
    total_connections = context_summary.get("total_connections", 0)
    
    overview = f"""
# Project Overview

## Project Root
{root}

## Project Statistics
- Total files analyzed: {total_files}
- Total connections between files: {total_connections}

## Key Files
"""
    
    if most_connected_files:
        overview += "### Most Connected Files\n"
        for file in most_connected_files[:5]:  # Top 5 most connected files
            overview += f"- {file.get('file', 'Unknown')} ({file.get('connections', 0)} connections)\n"
    
    if top_imported_modules:
        overview += "\n### Most Imported Modules\n"
        for module in top_imported_modules[:5]:  # Top 5 most imported modules
            overview += f"- {module.get('module', 'Unknown')} (imported {module.get('imported_by_count', 0)} times)\n"
    
    return overview

def generate_file_details(report: Dict[str, Any]) -> str:
    """
    Generate detailed information about each file in the project.
    
    Args:
        report: The CodeViz analysis report
        
    Returns:
        A string containing details about each file
    """
    summaries = report.get("summaries", [])
    interconnections = report.get("interconnections", {})
    
    if not summaries:
        return "No file summaries available."
    
    details = "\n# File Details\n\n"
    
    for summary in summaries:
        path = summary.get("path", "Unknown")
        language = summary.get("language", "Unknown")
        functions = summary.get("functions", [])
        classes = summary.get("classes", [])
        dependencies = summary.get("dependencies", [])
        size = summary.get("size", 0)
        line_count = summary.get("line_count", 0)
        
        # Get files that import this file
        imported_by = []
        for file, deps in interconnections.items():
            if path in deps:
                imported_by.append(file)
        
        details += f"## {path}\n"
        details += f"- **Language**: {language}\n"
        details += f"- **Size**: {size} bytes\n"
        details += f"- **Line Count**: {line_count}\n"
        
        if classes:
            details += f"- **Classes**:\n"
            for cls in classes:
                details += f"  - `{cls}`\n"
        
        if functions:
            details += f"- **Functions**:\n"
            for func in functions:
                details += f"  - `{func}`\n"
        
        if dependencies:
            details += f"- **Imports**:\n"
            for dep in dependencies:
                details += f"  - `{dep}`\n"
        
        if imported_by:
            details += f"- **Imported By**:\n"
            for imp in imported_by:
                details += f"  - `{imp}`\n"
        
        details += "\n"
    
    return details

def generate_dependency_graph(report: Dict[str, Any]) -> str:
    """
    Generate a textual representation of the dependency graph.
    
    Args:
        report: The CodeViz analysis report
        
    Returns:
        A string containing the dependency graph
    """
    interconnections = report.get("interconnections", {})
    
    if not interconnections:
        return "No dependency information available."
    
    graph = "\n# Dependency Graph\n\n"
    graph += "```\n"
    
    # Create a simple text-based graph
    for file, deps in interconnections.items():
        if deps:
            graph += f"{file} → {', '.join(deps)}\n"
        else:
            graph += f"{file} (no dependencies)\n"
    
    graph += "```\n"
    return graph

def generate_prompt(report: Dict[str, Any]) -> str:
    """
    Generate a complete prompt based on the report.
    
    Args:
        report: The CodeViz analysis report
        
    Returns:
        A string containing the complete prompt
    """
    prompt = """# CodeViz Analysis Report

This is an analysis of a codebase generated by CodeViz. The report includes information about the project structure, file details, and dependencies between files.

"""
    
    prompt += generate_project_overview(report)
    prompt += generate_dependency_graph(report)
    prompt += generate_file_details(report)
    
    prompt += """
# Task

Based on the above analysis of the codebase, please provide:

1. A high-level overview of the project's architecture
2. The main components and their responsibilities
3. The flow of data through the system
4. Any potential issues or areas for improvement
5. Suggestions for further development or refactoring

Feel free to ask questions if you need more information about specific parts of the codebase.
"""
    
    return prompt

def main():
    """Main function to parse arguments and generate the prompt."""
    if len(sys.argv) < 2:
        print("Usage: python report_to_prompt.py <report_path> [output_path]", file=sys.stderr)
        sys.exit(1)
    
    report_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "codebase_prompt.md"
    
    # Load the report
    report = load_report(report_path)
    
    # Generate the prompt
    prompt = generate_prompt(report)
    
    # Write the prompt to a file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Prompt successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing prompt to file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
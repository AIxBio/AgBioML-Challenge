"""Lab notebook functionality for autobioml."""

import os
import datetime
from pathlib import Path
from typing import Optional

from autogen_core.tools import FunctionTool


def initialize_notebook(notebook_path: Path) -> None:
    """
    Initialize the lab notebook file.
    
    Args:
        notebook_path: Path to the notebook file
    """
    # If the notebook doesn't exist, create it with a header
    if not notebook_path.exists():
        with open(notebook_path, 'w') as f:
            content = f"""# Lab Notebook
Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This notebook contains the record of experiments, decisions, and results for the current task.

## Project Information
This project uses an autonomous workflow where agents determine the best course
of action based on current project state. Agents are expected to:
1. Continuously assess project status through this notebook
2. Identify gaps and opportunities in the current research
3. Propose logical next steps based on scientific merit
4. Document all decisions, findings, and observations

## Entries

<!-- Entries will go here -->

"""
            f.write(content)


def read_notebook(notebook_path: Optional[Path] = None) -> str:
    """
    Read the entire lab notebook content.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        The entire content of the notebook
    """
    NOTEBOOK_CHAR_LIMIT = 100_000
    
    if notebook_path is None:
        notebook_path = Path("lab_notebook.md")
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
            # Truncate if content is too long
            if len(content) > NOTEBOOK_CHAR_LIMIT:
                print(f"WARNING: Lab notebook content exceeds {NOTEBOOK_CHAR_LIMIT} characters, truncating...")
                content = content[-NOTEBOOK_CHAR_LIMIT:]
            return content
    except FileNotFoundError:
        # Initialize the notebook if it doesn't exist
        initialize_notebook(notebook_path)
        return read_notebook(notebook_path)


def write_notebook(
    entry: str,
    entry_type: str = "NOTE",
    source: str = "SYSTEM",
    notebook_path: Optional[Path] = None
) -> str:
    """
    Append an entry to the lab notebook.
    
    Args:
        entry: The content to append to the notebook
        entry_type: Type of entry (e.g., NOTE, PLAN, OUTPUT)
        source: Name of agent that wrote the entry
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        The entry that was appended, with metadata
    """
    if notebook_path is None:
        notebook_path = Path("lab_notebook.md")
    
    # Ensure notebook exists
    if not notebook_path.exists():
        initialize_notebook(notebook_path)
    
    # Format the entry with metadata
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_entry = f"\n### [{timestamp}] {source} - {entry_type}\n\n{entry}\n\n"
    
    # Append the entry to the notebook
    with open(notebook_path, 'a') as f:
        f.write(formatted_entry)
    
    return formatted_entry


def read_notebook_recent(notebook_path: Optional[Path] = None, max_entries: int = 10) -> str:
    """
    Read only the most recent entries from the lab notebook.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
        max_entries: Maximum number of recent entries to return
    
    Returns:
        Content of the most recent entries only
    """
    if notebook_path is None:
        notebook_path = Path("lab_notebook.md")
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        # Split by entry headers (### [timestamp])
        parts = content.split("### [")
        
        if len(parts) <= max_entries + 1:  # +1 because first part is header
            return content
        
        # Keep the header (first part) and last N entries
        header = parts[0]
        recent_entries = parts[-(max_entries):]
        
        # Reconstruct with entry markers
        result = header + "### [" + "### [".join(recent_entries)
        
        return result
        
    except FileNotFoundError:
        # Initialize the notebook if it doesn't exist
        initialize_notebook(notebook_path)
        return read_notebook_recent(notebook_path, max_entries)


def read_notebook_summary(notebook_path: Optional[Path] = None, max_chars: int = 20000) -> str:
    """
    Read notebook content with smart truncation.
    
    Args:
        notebook_path: Path to the notebook file
        max_chars: Maximum characters to return
    
    Returns:
        Truncated notebook content with summary if needed
    """
    full_content = read_notebook(notebook_path)
    
    if len(full_content) <= max_chars:
        return full_content
    
    # Try recent entries first
    recent_content = read_notebook_recent(notebook_path, max_entries=8)
    if len(recent_content) <= max_chars:
        return recent_content
    
    # If still too long, truncate and add summary
    truncated = recent_content[-max_chars:]
    return f"""[NOTEBOOK CONTENT TRUNCATED - showing last {max_chars} characters]

{truncated}

[END TRUNCATED CONTENT]"""


# Create tool wrappers
notebook_read_tool = FunctionTool(
    read_notebook,
    name="read_notebook",
    description="Read the entire content of the lab notebook to access the team's progress, decisions, and results.",
)

notebook_write_tool = FunctionTool(
    write_notebook,
    name="write_notebook",
    description="""Append an entry to the lab notebook. 
    Parameters:
    - entry: The content to append to the notebook
    - entry_type: Type of entry (e.g., NOTE, PLAN, OUTPUT)
    - source: Source of the entry (your agent name)
    Always use this tool to document important decisions, specifications, results, or observations.
    """,
) 
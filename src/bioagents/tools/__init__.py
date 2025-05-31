"""Tools module for bioagents framework."""

from .notebook import (
    initialize_notebook,
    read_notebook,
    write_notebook,
    read_notebook_summary,
    notebook_read_tool,
    notebook_write_tool
)
from .research import get_research_tools
from .evaluation import get_evaluation_tools, set_evaluator, _evaluator
from .base import get_base_tools

__all__ = [
    "initialize_notebook",
    "read_notebook",
    "write_notebook",
    "read_notebook_summary",
    "get_research_tools",
    "get_evaluation_tools",
    "set_evaluator",
    "get_all_tools",
    "get_base_tools"
]


def get_all_tools():
    """Get all available tools."""
    tools = {}
    
    # Add base tools
    tools.update(get_base_tools())
    
    # Add notebook tools
    tools["read_notebook"] = notebook_read_tool
    tools["write_notebook"] = notebook_write_tool
    
    # Add research tools
    tools.update(get_research_tools())
    
    # Add evaluation tools
    tools.update(get_evaluation_tools())
    
    return tools

"""Tools package for bioagents framework."""

from .base import get_base_tools
from .notebook import notebook_read_tool, notebook_write_tool, initialize_notebook, read_notebook, write_notebook
from .research import get_research_tools

__all__ = [
    "get_all_tools",
    "get_base_tools", 
    "get_research_tools",
    "notebook_read_tool",
    "notebook_write_tool",
    "initialize_notebook",
    "read_notebook",
    "write_notebook"
]


def get_all_tools():
    """Get dictionary of all available tool instances."""
    tools = {}
    
    # Add base tools
    tools.update(get_base_tools())
    
    # Add notebook tools
    tools["read_notebook"] = notebook_read_tool
    tools["write_notebook"] = notebook_write_tool
    
    # Add research tools
    tools.update(get_research_tools())
    
    return tools

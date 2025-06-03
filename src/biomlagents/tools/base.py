"""Base tools for biomlagents framework."""

import os
import glob
import math
import datetime
import numpy as np
from typing import Union, Dict, List, Any
from pathlib import Path

from autogen_core.tools import FunctionTool
from PIL import Image as PILImage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_agentchat.agents import AssistantAgent


async def calculator(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression or perform a calculation.
    
    This tool can handle:
    - Basic arithmetic operations (+, -, *, /, **, %)
    - Mathematical functions (sqrt, sin, cos, tan, log, etc.)
    - Statistics (mean, median, std, etc.)
    - Weighted averages
    - Basic matrix operations
    
    Args:
        expression: A string containing a mathematical expression to evaluate
        
    Returns:
        A dictionary containing the result and additional context
    """
    try:
        # Create a safe environment with math functions but no builtins
        safe_env = {
            # Basic math functions
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
            # Math module functions
            'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log, 'log10': math.log10,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'degrees': math.degrees, 'radians': math.radians,
            'pi': math.pi, 'e': math.e,
            # NumPy functions for arrays and statistics
            'np': np, 'array': np.array, 'mean': np.mean, 'median': np.median,
            'std': np.std, 'var': np.var, 'percentile': np.percentile
        }
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, safe_env)
        
        # Format the response
        if isinstance(result, (np.ndarray, list)):
            # For array-like results, convert to list for JSON serialization
            result_value = result.tolist() if isinstance(result, np.ndarray) else result
            return {
                "result": result_value,
                "type": "array",
                "shape": np.array(result).shape if hasattr(np.array(result), "shape") else len(result),
                "expression": expression
            }
        else:
            # For scalar results
            return {
                "result": float(result) if isinstance(result, (int, float, np.number)) else result,
                "type": "scalar",
                "expression": expression
            }
    except Exception as e:
        return {
            "error": str(e),
            "type": "error",
            "expression": expression,
            "help": """
            Examples of valid expressions:
            - Basic arithmetic: "2 + 2 * 3" or "10 / 2 - 3"
            - Math functions: "sqrt(16)" or "sin(radians(30))"
            - Statistics: "mean([1, 2, 3, 4, 5])" or "median([1, 2, 3, 4, 5])"
            - IMPORTANT: Aggregator functions (sum, min, max) require iterables: "sum([1, 2, 3])" NOT "sum(1, 2, 3)"
            - Weighted average: "sum([0.05*3, 0.1*2, 0.2*3, 0.15*2, 0.05*3, 0.15*2, 0.1*3, 0.1*2, 0.05*3, 0.05*2])"
            """
        }


async def search_directory(directory_path: str, pattern: str = None, recursive: bool = False) -> str:
    """
    Search for files in a specified directory, optionally filtering by pattern and searching recursively.
    
    Args:
        directory_path: Path to the directory to search
        pattern: Optional glob pattern to filter results (e.g., "*.png" for all PNG files)
        recursive: Whether to search subdirectories recursively
        
    Returns:
        A string containing the list of matching files and their details
    """
    try:
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
            
        # Construct search pattern
        search_path = os.path.join(directory_path, pattern or "*")
        
        # Find files matching the pattern
        if recursive:
            # For recursive search
            matches = []
            if pattern:
                for root, _, _ in os.walk(directory_path):
                    matches.extend(glob.glob(os.path.join(root, pattern)))
            else:
                for root, _, files in os.walk(directory_path):
                    matches.extend([os.path.join(root, file) for file in files])
        else:
            # For non-recursive search
            matches = glob.glob(search_path)
        
        # Sort results
        matches.sort()
        
        # Format the output
        if not matches:
            if pattern:
                return f"No files matching '{pattern}' found in '{directory_path}'."
            else:
                return f"No files found in '{directory_path}'."
        
        result = f"Found {len(matches)} files in '{directory_path}'"
        if pattern:
            result += f" matching '{pattern}'"
        result += ":\n\n"
        
        # Add file details
        for filepath in matches:
            filename = os.path.basename(filepath)
            size = os.path.getsize(filepath)
            mod_time = os.path.getmtime(filepath)
            
            # Format size in human-readable format
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
                
            # Format modified time
            mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to result
            result += f"- {filename} ({size_str}, modified: {mod_time_str})\n"
            
        return result
        
    except Exception as e:
        return f"Error searching directory: {str(e)}"


async def read_text_file(filepath: str) -> str:
    """
    Read the contents of a text file.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        Contents of the file or error message
    """
    CHARACTER_LIMIT = 10_000
    try:
        with open(filepath, 'r') as file:
            content = file.read()
            if len(content) > CHARACTER_LIMIT:
                return content[:CHARACTER_LIMIT] + '... (truncated due to character limit in output of read_text_file)'
            else:
                return content
    except Exception as e:
        return f"Error reading text file: {str(e)}"


async def write_text_file(filepath: str, content: str) -> str:
    """
    Write the contents of a text file.
    
    Args:
        filepath: Path to the text file
        content: Content to write to the file
        
    Returns:
        Success message or error
    """
    try:
        with open(filepath, 'w') as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}"
    except Exception as e:
        return f"Error writing text file: {str(e)}"


async def read_arrow_file(filepath: str) -> str:
    """
    Read the contents of an Arrow file (feather format).
    
    Args:
        filepath: Path to the arrow file
        
    Returns:
        String representation of the dataframe or error message
    """
    import pandas as pd
    try:
        df = pd.read_feather(filepath)
        # If shape of any dimension is > 10, return the first 10 rows / columns
        orig_shape = df.shape
        if df.shape[0] > 10:
            df = df.head(10)
        if df.shape[1] > 10:
            df = df.iloc[:, :10]
        return df.to_string() + f"\n\n(Truncated due to character limit in output of read_arrow_file). Original data shape: {orig_shape}"
    except Exception as e:
        return f"Error reading Arrow file: {str(e)}"


async def analyze_plot_file(filepath: str, prompt: str | None = None) -> str:
    """
    Analyze a plot file and return a description of its contents.
    
    Args:
        filepath: Path to the plot file (relative to the working directory)
        prompt: Optional custom prompt to ask about the plot. If None, uses a default prompt.
        
    Returns:
        A string containing the analysis of the plot or an error message
    """
    try:
        # Initialize the model client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini"
        )

        # Create the agent that can handle multimodal input
        agent = AssistantAgent(
            name="plot_analyzer",
            model_client=model_client,
            system_message="""You are an agent that can analyze and describe plots.
            When given a plot, you should:
            1. Describe what type of plot it is
            2. Explain what the plot shows
            3. Identify key features of the distribution
            4. Note any interesting patterns or outliers
            5. Provide a clear summary of the insights
            6. Critically evaluate how well- or poorly-formed the plot is

            Rubric for evaluating plot quality:
            | **Criteria**                  | **Good**                                                            | **Fair**                                                              | **Poor**                                                              |
            |-------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
            | **Plot Type Appropriateness** | Plot type is well-suited to the data and clearly conveys its message. | Plot type is somewhat appropriate but may lead to minor confusion.    | Plot type is poorly chosen, causing significant misinterpretation.    |
            | **Data Integrity & Accuracy** | Data are accurately represented with proper scaling and minimal errors. | Minor inaccuracies or scaling issues are present.                     | Data are significantly misrepresented or distorted.                   |
            | **Clarity & Readability**     | All elements (labels, legends, etc.) are clear, legible, and organized.  | Some elements are hard to read or slightly cluttered.                   | The plot is cluttered with illegible or missing text elements.          |
            | **Self-Containment & Utility**| Plot includes all necessary details (titles, labels, legends) for stand-alone understanding. | Key details are missing, requiring some effort to grasp the plot's intent. | Essential information is absent, leaving the viewer confused.         |
            | **Overall Visual Quality**    | Clean design that focuses on clear data communication.               | Visual distractions are present but do not severely hinder understanding. | Distracting design elements significantly impair data communication.  |
            
            Your analysis should be thorough but concise. Give the score (Good, Fair, Poor) for each criterion and explain why.
            
            If given specific questions about the plot, answer them directly and clearly.
            
            Your response should always be in this format:

            **OVERALL EXPLANATION OF PLOT**
            [OVERALL EXPLANATION OF PLOT with analysis of what the plot shows]
            
            **CRITIQUE OF PLOT**
            [CRITIQUE OF PLOT with score (Good, Fair, Poor) for each criterion and explanation]

            **QUESTIONS AND ANSWERS ABOUT THE PLOT**
            Question: [QUESTION]
            Answer: [ANSWER]
            """,
            model_client_stream=True
        )

        # Load the plot as a PIL Image
        plot_image = PILImage.open(filepath)

        # Use default prompt if none provided
        if prompt is None:
            prompt = """Please analyze this plot and describe what it shows. Focus on:
            1. The type of plot and its purpose
            2. The distribution characteristics and patterns
            3. Any notable outliers or anomalies
            4. The approximate range of values
            5. Any insights that could be relevant for data analysis"""

        # Create a multimodal message with both text and the plot image
        message = MultiModalMessage(
            content=[
                prompt,
                Image(plot_image)
            ],
            source="user"
        )

        # Get the response
        response = await agent.on_messages(
            messages=[message],
            cancellation_token=None
        )

        # Close the model client
        await model_client.close()

        # Return the analysis
        return response.chat_message.content
        
    except Exception as e:
        return f"Error analyzing plot file: {str(e)}"


# Create tool instances
def get_base_tools():
    """Get dictionary of all base tool instances."""
    return {
        "calculator": FunctionTool(
            calculator,
            description="""Calculate mathematical expressions, statistics, or weighted averages.
            This tool can evaluate arithmetic, trigonometric functions, logs, and statistical operations.
            Very useful for calculating weighted scores in rubrics, statistical metrics, and other numerical analyses.
            
            IMPORTANT: Aggregator functions (sum, min, max) require iterables like lists.
            Always use: sum([value1, value2, ...]) NOT sum(value1, value2, ...)
            
            Example expressions:
            - Basic: "2 + 2 * 3"
            - Functions: "sqrt(16)" or "sin(radians(30))"
            - Statistics: "mean([1, 2, 3, 4, 5])"
            - Weighted average: "sum([0.05*3, 0.1*2, 0.2*3, 0.15*2, 0.05*3, 0.15*2, 0.1*3, 0.1*2, 0.05*3, 0.05*2])"
            """,
            name="calculator"
        ),
        "search_directory": FunctionTool(
            search_directory,
            description="Search for files in a specified directory, optionally filtering by pattern and searching recursively.",
            name="search_directory"
        ),
        "read_text_file": FunctionTool(
            read_text_file,
            description="Read the contents of a text file (.txt, .csv, .tsv, .json, etc.). Only returns the first 10,000 characters of the file.",
            name="read_text_file"
        ),
        "write_text_file": FunctionTool(
            write_text_file,
            description="Write the contents of a text file (.txt, .csv, .tsv, .json, etc.).",
            name="write_text_file"
        ),
        "read_arrow_file": FunctionTool(
            read_arrow_file,
            description="Read the contents of an Arrow file (feather format) as a pandas DataFrame. Only returns the head of the DataFrame. Useful for examining the first few rows of a dataset.",
            name="read_arrow_file"
        ),
        "analyze_plot": FunctionTool(
            analyze_plot_file,
            description="""Analyze and critique plot files using advanced visual AI analysis.
            
            This tool provides detailed analysis that goes beyond what you can determine from code alone:
            - Identifies plot types, data patterns, and visual quality issues
            - Detects outliers, trends, and distribution characteristics visible in the plot
            - Provides expert critique of plot design, clarity, and scientific communication
            - Can answer specific questions about visual elements you can't assess programmatically
            
            **WHEN TO USE:**
            - After creating any visualization to verify it displays data correctly
            - To get expert feedback on plot quality and design before finalizing
            - To identify visual patterns or outliers that might not be obvious from code
            - To ensure plots meet scientific communication standards
            - When you need to describe plots in reports or documentation
            
            **USAGE EXAMPLES:**
            - analyze_plot("correlation_heatmap.png") - Get detailed analysis of patterns
            - analyze_plot("age_distribution.png", "Are there any obvious outliers in this distribution?")
            - analyze_plot("model_performance.png", "How clear is this plot for showing model comparison?")
            
            **PARAMETERS:**
            - filepath: Path to the plot file (PNG, JPG, etc.)
            - prompt: Optional specific question about the plot (if not provided, gives comprehensive analysis)
            
            The tool returns structured analysis including plot explanation, quality critique, and answers to questions.""",
            name="analyze_plot"
        ),
    } 
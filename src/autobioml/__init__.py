"""autobioml - A framework for autonomous biomedical ML research agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env file from current directory or parent directories
load_dotenv()

__version__ = "0.1.0"

from .cli import main

__all__ = ["main"]

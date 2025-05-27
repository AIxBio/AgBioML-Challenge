"""Setup script for bioagents package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bioagents",
    version="0.1.0",
    author="BioAgents Team",
    description="A framework for autonomous biomedical ML research agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bioagents",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "autogen-agentchat>=0.4.0",
        "autogen-ext[openai]>=0.4.0",
        "hydra-core>=1.3.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "docker>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bioagents=bioagents.cli:main",
        ],
    },
    package_data={
        "bioagents": ["config/*.yaml"],
    },
) 
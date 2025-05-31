# BioAgents Challenges

This directory contains biomedical machine learning challenges for the BioAgents framework. Each challenge follows a standardized structure to ensure compatibility with the autonomous agent system.

## Challenge Structure

Every challenge must follow this directory structure:

```
challenge_name/
├── task.yaml          # Task configuration (REQUIRED)
├── data/
│   ├── agent/         # Data visible to agents during execution
│   └── eval/          # Held-out evaluation data (not visible to agents)
├── scripts/
│   └── evaluate.py    # Evaluation script for the challenge
├── Dockerfile         # Task-specific Docker image (optional, uses default if not provided)
└── README.md          # Human-readable description of the challenge
```

## Creating a New Challenge

### 1. Task Configuration (task.yaml)

The `task.yaml` file defines the challenge and must include:

```yaml
name: challenge_identifier  # Alphanumeric + underscores only
display_name: Human Readable Challenge Name
version: "1.0"  # Format: X.Y

task_description: |
  Comprehensive description of what agents need to accomplish.
  
project_goal: |
  Specific, measurable objectives for the task.
  
project_context: |
  Background information and domain knowledge.

available_data:
  agent_data:
    - path: data/agent/file1.arrow
      description: Description of the data file
  eval_data:
    - path: data/eval/file1.arrow
      description: Description of evaluation data

data_completeness: |
  Statement about data sufficiency and any important notes.

evaluation:
  metrics:
    - name: metric_name
      threshold: 0.9
      dataset: test_set
  required_outputs:
    - model_checkpoint
    - inference_script
    - evaluation_results
    - analysis_report

docker:
  gpu_required: false
  base_image: millerh1/bioagents:latest
  additional_packages: null
```

### 2. Data Organization

#### Agent Data (`data/agent/`)
- Data that agents can access during execution
- Should be in Arrow (Parquet) format for efficiency
- Include all necessary files for the task
- Use descriptive filenames

#### Evaluation Data (`data/eval/`)
- Held-out test data for final evaluation
- Not accessible to agents during execution
- Used by the evaluation script to assess performance
- Should follow the same format as agent data

### 3. Evaluation Script (`scripts/evaluate.py`)

Create an evaluation script that:
- Loads predictions from the agent's output
- Compares against held-out evaluation data
- Calculates all metrics specified in task.yaml
- Outputs results in a standardized format

Example template:

```python
#!/usr/bin/env python
"""Evaluation script for [challenge name]."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

def evaluate(predictions_path: Path, eval_data_path: Path) -> Dict[str, Any]:
    """
    Evaluate predictions against held-out test data.
    
    Args:
        predictions_path: Path to predictions file
        eval_data_path: Path to evaluation data directory
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load predictions and ground truth
    # Calculate metrics
    # Return results
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions")
    parser.add_argument("--eval-data", required=True, help="Path to eval data")
    parser.add_argument("--output", required=True, help="Output path for results")
    
    args = parser.parse_args()
    
    results = evaluate(Path(args.predictions), Path(args.eval_data))
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
```

### 4. Dockerfile (Optional)

If your challenge requires specific dependencies beyond the base image:

```dockerfile
FROM millerh1/bioagents:latest

# Install additional dependencies
RUN pip install special-package==1.0.0

# Copy any challenge-specific resources
COPY resources/ /opt/resources/
```

### 5. Challenge README

Create a `README.md` with:
- Challenge overview and motivation
- Data description and sources
- Evaluation criteria and metrics
- Example solutions or baselines
- References and citations

## Available Challenges

1. **01_basic_epigenetic_clock** - Develop an epigenetic age prediction model
2. **02_rsna_breast_cancer_detection** - Detect breast cancer from mammography images
3. **03_tdc_admet_benchmark** - Predict ADMET properties for drug discovery
4. **04_cellmap_challenge** - Cell type classification from single-cell data
5. **05_dream_target_2035** - Drug target discovery challenge
6. **06_biomarkers_of_aging** - Identify biomarkers of aging from multi-omics data

## Testing Your Challenge

Before submitting, test your challenge:

```bash
# Validate configuration
bioagents task_dir=challenges/your_challenge dry_run=true

# Run with minimal iterations for testing
bioagents task_dir=challenges/your_challenge max_iterations=2

# Test evaluation script
python challenges/your_challenge/scripts/evaluate.py \
    --predictions sample_predictions.json \
    --eval-data challenges/your_challenge/data/eval/ \
    --output results.json
```

## Best Practices

1. **Data Format**: Use Arrow/Parquet format for large datasets
2. **Clear Objectives**: Define specific, measurable success criteria
3. **Comprehensive Documentation**: Include all necessary domain knowledge
4. **Reproducibility**: Ensure evaluation can be run consistently
5. **Resource Constraints**: Consider computational requirements
6. **Version Control**: Use semantic versioning for updates

## Contributing

To contribute a new challenge:

1. Follow the structure outlined above
2. Ensure all required files are present
3. Test thoroughly with the framework
4. Submit a pull request with your challenge

For questions or support, please open an issue in the repository.

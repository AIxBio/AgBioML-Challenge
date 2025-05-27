#!/usr/bin/env python
"""Script to create a new challenge template for BioAgents."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any


def create_task_yaml(challenge_name: str, display_name: str) -> str:
    """Create a template task.yaml file."""
    return f"""name: {challenge_name}
display_name: {display_name}
version: "1.0"

task_description: |
  [Describe what agents need to accomplish in this challenge.
   Be specific about the problem, expected inputs, and desired outputs.]

project_goal: |
  [Define specific, measurable objectives for the task.
   What constitutes success? What metrics will be used?]

project_context: |
  [Provide background information and domain knowledge.
   What should agents know about this problem domain?]

available_data:
  agent_data:
    - path: data/agent/training_data.arrow
      description: Main training dataset
    - path: data/agent/metadata.arrow
      description: Sample metadata and labels
  eval_data:
    - path: data/eval/test_data.arrow
      description: Held-out test dataset
    - path: data/eval/test_metadata.arrow
      description: Test set metadata and labels

data_completeness: |
  [Explain data sufficiency and any important notes about the dataset.
   Are there any limitations or special considerations?]

autonomous_workflow:
  approach: |
    [High-level guidance on how agents should approach this task.
     What are the key steps in solving this problem?]
  methodology: |
    [Suggested methodological considerations.
     What techniques or approaches are most relevant?]
  expected_outcomes:
    - Exploratory data analysis with visualizations
    - Trained machine learning model
    - Performance evaluation results
    - Analysis report with findings

lab_notebook_guidelines: |
  [Instructions for what should be documented in the lab notebook.
   What decisions, experiments, and results should be recorded?]

reference:
  background:
    text: |
      [Reference information agents might need.
       Include relevant papers, methods, or domain knowledge.]
  evaluation_metrics:
    text: |
      [Explanation of evaluation metrics and their interpretation.
       How should agents assess their model performance?]

evaluation:
  metrics:
    - name: primary_metric
      threshold: 0.8
      dataset: test_set
    - name: secondary_metric
      threshold: 0.1
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
"""


def create_evaluation_script(challenge_name: str) -> str:
    """Create a template evaluation script."""
    return f'''#!/usr/bin/env python
"""Evaluation script for the {challenge_name} challenge."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load predictions from various possible formats.
    
    Args:
        predictions_path: Path to predictions file
        
    Returns:
        DataFrame with predictions
    """
    if predictions_path.suffix == '.arrow':
        return pd.read_feather(predictions_path)
    elif predictions_path.suffix == '.csv':
        return pd.read_csv(predictions_path)
    elif predictions_path.suffix == '.json':
        with open(predictions_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {{predictions_path.suffix}}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        
    Returns:
        Dictionary of metrics
    """
    # TODO: Implement your evaluation metrics here
    # Examples:
    # - Classification: accuracy, precision, recall, F1, AUC
    # - Regression: MAE, RMSE, R², correlation
    # - Custom metrics specific to your domain
    
    return {{
        'primary_metric': 0.0,  # Replace with actual calculation
        'secondary_metric': 0.0,  # Replace with actual calculation
        'n_samples': len(y_true)
    }}


def evaluate(predictions_path: Path, eval_data_path: Path) -> Dict[str, Any]:
    """
    Evaluate predictions against held-out test data.
    
    Args:
        predictions_path: Path to predictions file
        eval_data_path: Path to evaluation data directory
        
    Returns:
        Dictionary of evaluation results
    """
    # Load predictions
    predictions = load_predictions(predictions_path)
    
    # Load ground truth
    # TODO: Update these paths based on your eval data structure
    test_data_path = eval_data_path / 'test_data.arrow'
    test_metadata_path = eval_data_path / 'test_metadata.arrow'
    
    if not test_metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {{test_metadata_path}}")
    
    test_metadata = pd.read_feather(test_metadata_path)
    
    # Merge predictions with ground truth
    # TODO: Update merge logic based on your data structure
    merged = test_metadata.merge(predictions, on='sample_id', how='inner')
    
    if len(merged) == 0:
        raise ValueError("No matching samples found between predictions and ground truth")
    
    # Calculate metrics
    # TODO: Update based on your specific task (classification vs regression)
    overall_metrics = calculate_metrics(
        merged['true_label'].values,  # Update column name
        merged['predicted_label'].values  # Update column name
    )
    
    # Check if performance criteria are met
    criteria_met = {{
        'primary_metric >= threshold': overall_metrics['primary_metric'] >= 0.8,  # Update threshold
        'secondary_metric <= threshold': overall_metrics['secondary_metric'] <= 0.1,  # Update threshold
    }}
    
    return {{
        'overall_metrics': overall_metrics,
        'criteria_met': criteria_met,
        'all_criteria_passed': all(criteria_met.values()),
        'evaluation_summary': {{
            'total_samples': len(test_metadata),
            'evaluated_samples': len(merged),
            'missing_predictions': len(test_metadata) - len(merged)
        }}
    }}


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate {challenge_name} predictions')
    parser.add_argument('--predictions', required=True, help='Path to predictions file')
    parser.add_argument('--eval-data', required=True, help='Path to evaluation data directory')
    parser.add_argument('--output', required=True, help='Output path for results')
    
    args = parser.parse_args()
    
    try:
        results = evaluate(
            Path(args.predictions),
            Path(args.eval_data)
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\\n=== Evaluation Results ===")
        print(f"Primary Metric: {{results['overall_metrics']['primary_metric']:.4f}}")
        print(f"Secondary Metric: {{results['overall_metrics']['secondary_metric']:.4f}}")
        print(f"Samples Evaluated: {{results['evaluation_summary']['evaluated_samples']}}")
        
        if results['all_criteria_passed']:
            print("\\n✓ All performance criteria met!")
        else:
            print("\\n✗ Some criteria not met:")
            for criterion, passed in results['criteria_met'].items():
                print(f"  {{'✓' if passed else '✗'}} {{criterion}}")
        
    except Exception as e:
        print(f"Evaluation failed: {{e}}")
        raise


if __name__ == "__main__":
    main()
'''


def create_readme(challenge_name: str, display_name: str) -> str:
    """Create a template README file."""
    return f"""# {display_name}

## Overview

[Provide a brief overview of the challenge and its significance in the biomedical domain.]

## Background

[Explain the scientific background and motivation for this challenge.
What problem does it address? Why is it important?]

## Challenge Description

[Detailed description of what agents need to accomplish:
- Input data format and structure
- Expected outputs
- Performance criteria
- Any special requirements]

## Data Description

### Training Data (`data/agent/`)

- **`training_data.arrow`**: [Description of main training data]
- **`metadata.arrow`**: [Description of metadata/labels]

### Evaluation Data (`data/eval/`)

- **`test_data.arrow`**: [Description of test data]
- **`test_metadata.arrow`**: [Description of test labels]

## Performance Criteria

[Define specific, measurable success criteria:
- Primary metrics and thresholds
- Secondary metrics
- Any additional requirements]

## Key Challenges

[List the main technical and scientific challenges:
1. Challenge 1
2. Challenge 2
3. Challenge 3]

## Evaluation

The evaluation script (`scripts/evaluate.py`) calculates:
- [List of metrics calculated]
- [Performance breakdowns by subgroups if applicable]

### Usage
```bash
python scripts/evaluate.py \\
    --predictions predictions.arrow \\
    --eval-data data/eval/ \\
    --output evaluation_results.json
```

## Expected Approach

[Outline the typical approach to solving this challenge:
1. Data exploration and preprocessing
2. Model development
3. Validation strategy
4. Final evaluation]

## Baseline Performance

[If available, provide baseline performance numbers or reference methods]

## References

[List relevant papers, datasets, or resources]

## Tips for Success

[Provide helpful tips for agents working on this challenge]
"""


def create_dockerfile(challenge_name: str) -> str:
    """Create a template Dockerfile."""
    return f"""FROM millerh1/bioagents:latest

# Install challenge-specific dependencies
# RUN pip install package-name==version

# Copy any challenge-specific resources
# COPY resources/ /opt/resources/

# Set working directory
WORKDIR /workspace

# Add any environment variables
# ENV CHALLENGE_NAME={challenge_name}
"""


def create_challenge(challenge_name: str, display_name: str, base_dir: Path) -> None:
    """Create a new challenge directory structure."""
    challenge_dir = base_dir / challenge_name
    
    # Create directory structure
    challenge_dir.mkdir(exist_ok=True)
    (challenge_dir / "data" / "agent").mkdir(parents=True, exist_ok=True)
    (challenge_dir / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (challenge_dir / "scripts").mkdir(exist_ok=True)
    
    # Create files
    files_to_create = {
        "task.yaml": create_task_yaml(challenge_name, display_name),
        "README.md": create_readme(challenge_name, display_name),
        "Dockerfile": create_dockerfile(challenge_name),
        "scripts/evaluate.py": create_evaluation_script(challenge_name),
    }
    
    for file_path, content in files_to_create.items():
        full_path = challenge_dir / file_path
        with open(full_path, 'w') as f:
            f.write(content)
        
        # Make evaluation script executable
        if file_path.endswith('evaluate.py'):
            os.chmod(full_path, 0o755)
    
    # Create placeholder data files
    data_placeholders = [
        "data/agent/training_data.arrow",
        "data/agent/metadata.arrow", 
        "data/eval/test_data.arrow",
        "data/eval/test_metadata.arrow"
    ]
    
    for placeholder in data_placeholders:
        placeholder_path = challenge_dir / placeholder
        with open(placeholder_path, 'w') as f:
            f.write("# Placeholder - replace with actual data file\\n")
    
    print(f"✓ Created challenge template: {challenge_dir}")
    print(f"✓ Directory structure created")
    print(f"✓ Template files generated")
    print(f"\\nNext steps:")
    print(f"1. Add your data files to data/agent/ and data/eval/")
    print(f"2. Update task.yaml with specific requirements")
    print(f"3. Implement evaluation logic in scripts/evaluate.py")
    print(f"4. Update README.md with challenge details")
    print(f"5. Test with: bioagents task_dir={challenge_dir} dry_run=true")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create a new BioAgents challenge template')
    parser.add_argument('challenge_name', help='Challenge identifier (alphanumeric + underscores)')
    parser.add_argument('display_name', help='Human-readable challenge name')
    parser.add_argument('--output-dir', default='challenges', help='Output directory (default: challenges)')
    
    args = parser.parse_args()
    
    # Validate challenge name
    if not args.challenge_name.replace('_', '').isalnum():
        print("Error: Challenge name must contain only alphanumeric characters and underscores")
        sys.exit(1)
    
    base_dir = Path(args.output_dir)
    base_dir.mkdir(exist_ok=True)
    
    create_challenge(args.challenge_name, args.display_name, base_dir)


if __name__ == "__main__":
    main() 
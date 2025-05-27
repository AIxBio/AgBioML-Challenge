#!/usr/bin/env python
"""
Test script to validate the evaluation workflow.

This script:
1. Loads the held-out test features from the agent directory
2. Creates dummy predictions
3. Saves them in the required format
4. Runs the evaluation script to ensure everything works
"""

import pandas as pd
import numpy as np
import subprocess
import json
from pathlib import Path
import sys


def test_workflow():
    """Test the complete evaluation workflow."""
    print("Testing evaluation workflow...")
    print("=" * 60)
    
    # Step 1: Load held-out test features
    print("\n1. Loading held-out test features...")
    try:
        betas_heldout = pd.read_feather('data/agent/betas_heldout.arrow')
        print(f"✓ Loaded {len(betas_heldout)} test samples")
        print(f"  Shape: {betas_heldout.shape}")
        
        # Get sample IDs
        sample_ids = betas_heldout.index
        print(f"  Sample IDs: {sample_ids[:5].tolist()}... (showing first 5)")
    except Exception as e:
        print(f"✗ Error loading test features: {e}")
        return False
    
    # Step 2: Create dummy predictions
    print("\n2. Creating dummy predictions...")
    # Create somewhat realistic predictions (random around 50 with some noise)
    base_age = 50
    noise = np.random.normal(0, 15, size=len(betas_heldout))
    predicted_ages = base_age + noise
    predicted_ages = np.clip(predicted_ages, 0, 100)  # Keep ages in reasonable range
    
    print(f"  Predicted age range: {predicted_ages.min():.1f} - {predicted_ages.max():.1f}")
    print(f"  Mean predicted age: {predicted_ages.mean():.1f}")
    
    # Step 3: Save predictions
    print("\n3. Saving predictions...")
    predictions_df = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_age': predicted_ages
    })
    
    try:
        predictions_df.to_feather('predictions.arrow')
        print("✓ Saved predictions.arrow")
        print(f"  Columns: {list(predictions_df.columns)}")
        print(f"  Data types: {predictions_df.dtypes.to_dict()}")
    except Exception as e:
        print(f"✗ Error saving predictions: {e}")
        return False
    
    # Step 4: Verify the file exists and can be loaded
    print("\n4. Verifying predictions file...")
    try:
        loaded_predictions = pd.read_feather('predictions.arrow')
        print("✓ Successfully loaded predictions.arrow")
        print(f"  Shape: {loaded_predictions.shape}")
        assert 'sample_id' in loaded_predictions.columns, "Missing 'sample_id' column"
        assert 'predicted_age' in loaded_predictions.columns, "Missing 'predicted_age' column"
        print("✓ Required columns present")
    except Exception as e:
        print(f"✗ Error verifying predictions: {e}")
        return False
    
    # Step 5: Run evaluation script
    print("\n5. Running evaluation script...")
    eval_command = [
        'python', 'scripts/evaluate.py',
        '--predictions', 'predictions.arrow',
        '--eval-data', 'data/eval',
        '--output', 'test_evaluation_results.json'
    ]
    
    try:
        result = subprocess.run(eval_command, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Evaluation completed successfully")
            print("\nEvaluation output:")
            print(result.stdout)
        else:
            print("✗ Evaluation failed")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running evaluation: {e}")
        return False
    
    # Step 6: Check evaluation results
    print("\n6. Checking evaluation results...")
    try:
        with open('test_evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        print("✓ Loaded evaluation results")
        print(f"  Overall Pearson correlation: {results['overall_metrics']['pearson_correlation']:.4f}")
        print(f"  Overall MAE: {results['overall_metrics']['mae_years']:.2f} years")
        print(f"  Samples evaluated: {results['evaluation_summary']['evaluated_samples']}")
        print(f"  All criteria passed: {results['all_criteria_passed']}")
        
        # Check if the number of evaluated samples matches expected
        if results['evaluation_summary']['evaluated_samples'] == len(betas_heldout):
            print("✓ All test samples were evaluated")
        else:
            print(f"⚠ Warning: Only {results['evaluation_summary']['evaluated_samples']}/{len(betas_heldout)} samples evaluated")
    
    except Exception as e:
        print(f"✗ Error checking results: {e}")
        return False
    
    # Cleanup
    print("\n7. Cleaning up test files...")
    try:
        Path('predictions.arrow').unlink(missing_ok=True)
        Path('test_evaluation_results.json').unlink(missing_ok=True)
        print("✓ Cleaned up test files")
    except Exception as e:
        print(f"⚠ Warning: Could not clean up some files: {e}")
    
    print("\n" + "=" * 60)
    print("✓ EVALUATION WORKFLOW TEST PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # Change to the challenge directory
    challenge_dir = Path(__file__).parent.parent
    import os
    os.chdir(challenge_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run the test
    success = test_workflow()
    sys.exit(0 if success else 1) 
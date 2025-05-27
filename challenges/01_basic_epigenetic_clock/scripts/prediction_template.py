#!/usr/bin/env python
"""
Template for generating predictions in the required format for evaluation.

This script demonstrates how to:
1. Load the held-out test features
2. Apply your trained model to generate predictions
3. Save predictions in the required format
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_predictions(sample_ids, predicted_ages, output_path='predictions.arrow'):
    """
    Save predictions in the required format for evaluation.
    
    Args:
        sample_ids: List or array of sample IDs from the held-out test set
        predicted_ages: List or array of predicted ages (must be same length as sample_ids)
        output_path: Path to save predictions (default: 'predictions.arrow')
    
    Returns:
        pd.DataFrame: The predictions dataframe that was saved
    """
    # Ensure inputs are the same length
    if len(sample_ids) != len(predicted_ages):
        raise ValueError(f"sample_ids ({len(sample_ids)}) and predicted_ages ({len(predicted_ages)}) must have the same length")
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'sample_id': sample_ids,
        'predicted_age': predicted_ages
    })
    
    # Ensure predicted_age is float type
    predictions_df['predicted_age'] = predictions_df['predicted_age'].astype(float)
    
    # Save to arrow format
    predictions_df.to_feather(output_path)
    
    print(f"✓ Predictions saved to {output_path}")
    print(f"  Shape: {predictions_df.shape}")
    print(f"  Columns: {list(predictions_df.columns)}")
    print(f"  Age range: {predictions_df['predicted_age'].min():.1f} - {predictions_df['predicted_age'].max():.1f} years")
    print(f"  Mean age: {predictions_df['predicted_age'].mean():.1f} years")
    
    return predictions_df


def example_usage():
    """
    Example of how to use this template with your trained model.
    
    Replace this with your actual model loading and prediction code.
    """
    print("Loading held-out test features...")
    
    # Load the held-out test features
    betas_heldout = pd.read_feather('data/agent/betas_heldout.arrow')
    print(f"Loaded {len(betas_heldout)} test samples with {betas_heldout.shape[1]} features")
    
    # Get sample IDs (assuming they are the index)
    sample_ids = betas_heldout.index
    
    # TODO: Replace this section with your actual model prediction code
    # Example placeholder predictions (random ages between 20 and 80)
    print("\nGenerating predictions with your model...")
    # predicted_ages = your_model.predict(betas_heldout)
    predicted_ages = np.random.uniform(20, 80, size=len(betas_heldout))  # REPLACE THIS!
    
    # Save predictions in the required format
    print("\nSaving predictions...")
    predictions_df = save_predictions(sample_ids, predicted_ages)
    
    # Verify the file was created
    if Path('predictions.arrow').exists():
        print("\n✓ Success! predictions.arrow has been created.")
        print("  This file will be used for evaluation.")
    else:
        print("\n✗ Error: predictions.arrow was not created!")
    
    return predictions_df


if __name__ == "__main__":
    # Run the example
    example_usage()
    
    print("\n" + "="*60)
    print("IMPORTANT REMINDERS:")
    print("1. Replace the random predictions with your actual model predictions")
    print("2. Ensure you predict for ALL samples in betas_heldout.arrow")
    print("3. The output file MUST be named 'predictions.arrow'")
    print("4. The file MUST contain columns: 'sample_id' and 'predicted_age'")
    print("="*60) 
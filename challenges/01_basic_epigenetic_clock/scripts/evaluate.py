#!/usr/bin/env python
"""Evaluation script for the Basic Epigenetic Clock challenge."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load predictions from various possible formats.
    
    Args:
        predictions_path: Path to predictions file
        
    Returns:
        DataFrame with 'sample_id' and 'predicted_age' columns
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
        raise ValueError(f"Unsupported file format: {predictions_path.suffix}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for age prediction.
    
    Args:
        y_true: True ages
        y_pred: Predicted ages
        
    Returns:
        Dictionary of metrics
    """
    # Pearson correlation
    corr, p_value = pearsonr(y_true, y_pred)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Median Absolute Error
    median_ae = np.median(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'pearson_correlation': float(corr),
        'pearson_p_value': float(p_value),
        'mae_years': float(mae),
        'rmse_years': float(rmse),
        'median_ae_years': float(median_ae),
        'r_squared': float(r_squared),
        'n_samples': len(y_true)
    }


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
    metadata_path = eval_data_path / 'meta_heldout.arrow'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Evaluation metadata not found: {metadata_path}")
    
    metadata = pd.read_feather(metadata_path)
    
    # Reset index to make sample_id a column if it's in the index
    if metadata.index.name is not None or len(metadata.columns) < 5:
        metadata = metadata.reset_index()
        # Rename the index column to sample_id if needed
        if 'index' in metadata.columns:
            metadata = metadata.rename(columns={'index': 'sample_id'})
    
    # Ensure predictions has sample_id as a column
    if 'sample_id' not in predictions.columns:
        raise ValueError("Predictions must have a 'sample_id' column")
    
    # Merge predictions with ground truth
    merged = metadata.merge(predictions, on='sample_id', how='inner')
    
    if len(merged) == 0:
        raise ValueError("No matching samples found between predictions and ground truth")
    
    if len(merged) < len(metadata):
        print(f"Warning: Only {len(merged)}/{len(metadata)} samples have predictions")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(
        merged['age'].values,
        merged['predicted_age'].values
    )
    
    # Calculate metrics by tissue type if available
    tissue_metrics = {}
    if 'tissue_type' in merged.columns:
        for tissue in merged['tissue_type'].unique():
            tissue_data = merged[merged['tissue_type'] == tissue]
            if len(tissue_data) >= 10:  # Only calculate if sufficient samples
                tissue_metrics[tissue] = calculate_metrics(
                    tissue_data['age'].values,
                    tissue_data['predicted_age'].values
                )
    
    # Calculate metrics by age group
    age_groups = {
        'young': (0, 30),
        'middle': (30, 60),
        'old': (60, 100)
    }
    
    age_group_metrics = {}
    for group_name, (min_age, max_age) in age_groups.items():
        group_data = merged[(merged['age'] >= min_age) & (merged['age'] < max_age)]
        if len(group_data) >= 10:
            age_group_metrics[group_name] = calculate_metrics(
                group_data['age'].values,
                group_data['predicted_age'].values
            )
    
    # Check if performance criteria are met
    criteria_met = {
        'pearson_correlation >= 0.9': overall_metrics['pearson_correlation'] >= 0.9,
        'mae_years < 10.0': overall_metrics['mae_years'] < 10.0
    }
    
    return {
        'overall_metrics': overall_metrics,
        'tissue_metrics': tissue_metrics,
        'age_group_metrics': age_group_metrics,
        'criteria_met': criteria_met,
        'all_criteria_passed': all(criteria_met.values()),
        'evaluation_summary': {
            'total_samples': len(metadata),
            'evaluated_samples': len(merged),
            'missing_predictions': len(metadata) - len(merged)
        }
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate epigenetic clock predictions')
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
        print("\n=== Evaluation Results ===")
        print(f"Pearson Correlation: {results['overall_metrics']['pearson_correlation']:.4f}")
        print(f"MAE: {results['overall_metrics']['mae_years']:.2f} years")
        print(f"RMSE: {results['overall_metrics']['rmse_years']:.2f} years")
        print(f"Samples Evaluated: {results['evaluation_summary']['evaluated_samples']}")
        
        if results['all_criteria_passed']:
            print("\n✓ All performance criteria met!")
        else:
            print("\n✗ Some criteria not met:")
            for criterion, passed in results['criteria_met'].items():
                print(f"  {'✓' if passed else '✗'} {criterion}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 